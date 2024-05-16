from typing import Dict, List, Optional
import logging
from datetime import datetime
from llama_index import OpenAIEmbedding, ServiceContext, VectorStoreIndex
import nest_asyncio
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.callbacks.base import BaseCallbackHandler
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.node_parser import SentenceSplitter
from llama_index.callbacks.base import BaseCallbackHandler, CallbackManager
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingMode,
    OpenAIEmbeddingModelType,
)
from llama_index.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType,
)
from app.core.config import settings
from app.schema import (
    Message as MessageSchema,
    Document as DocumentSchema,
    Conversation as ConversationSchema,
    DocumentMetadataKeysEnum,
)
from app.chat.notebook_constants import (
    NODE_PARSER_CHUNK_OVERLAP,
    NODE_PARSER_CHUNK_SIZE,
    SYSTEM_MESSAGE,
)
from app.chat.tools import get_api_query_engine_tool
from app.chat.utils import build_title_for_document
from app.chat.qa_response_synth import get_custom_response_synth
from app.chat.engine import (
    build_description_for_document,
    get_chat_history,
    get_tool_service_context,
    index_to_query_engine,
)


logger = logging.getLogger(__name__)


logger.info("Applying nested asyncio patch")
nest_asyncio.apply()

OPENAI_TOOL_LLM_NAME = "gpt-3.5-turbo-0613"
OPENAI_CHAT_LLM_NAME = "gpt-3.5-turbo-0613"


async def get_chat_engine(
    callback_handler: BaseCallbackHandler,
    conversation: ConversationSchema,
    doc_id_to_index: Dict[str, VectorStoreIndex],
) -> OpenAIAgent:
    service_context = get_tool_service_context([callback_handler])
    # s3_fs = get_s3_fs()
    # doc_id_to_index = await build_doc_id_to_index_map(
    #     service_context, conversation.documents, fs=s3_fs
    # )
    id_to_doc: Dict[str, DocumentSchema] = {
        str(doc.id): doc for doc in conversation.documents
    }

    vector_query_engine_tools = [
        QueryEngineTool(
            # index.as_query_engine with MetadataFilters to we know this
            # is for node with the metadata db_document_id being equal doc_id
            query_engine=index_to_query_engine(doc_id, index),
            # metadata helps the agent to know if it has to use this QueryEngineTool or not.
            # Ex: if question about the company X, and QueryEngineTool is for the company X,
            # then agent know it has to use this QueryEngineTool.
            metadata=ToolMetadata(
                name=doc_id,
                description=build_description_for_document(id_to_doc[doc_id]),
            ),
        )
        for doc_id, index in doc_id_to_index.items()
    ]

    # response_synth = get_custom_response_synth(service_context, conversation.documents)

    # qualitative_question_engine = SubQuestionQueryEngine.from_defaults(
    #     query_engine_tools=vector_query_engine_tools,
    #     service_context=service_context,
    #     response_synthesizer=response_synth,
    #     verbose=settings.VERBOSE,
    #     use_async=True,
    # )

    #     top_level_sub_tools = [
    #         QueryEngineTool(
    #             query_engine=qualitative_question_engine,
    #             metadata=ToolMetadata(
    #                 name="qualitative_question_engine",
    #                 description="""
    # A query engine that can answer qualitative questions about a set of SEC financial documents that the user pre-selected for the conversation.
    # Any questions about company-related headwinds, tailwinds, risks, sentiments, or administrative information should be asked here.
    # """.strip(),
    #             ),
    #         )
    #     ]

    chat_llm = OpenAI(
        temperature=0,
        model=OPENAI_CHAT_LLM_NAME,
        streaming=True,
        api_key=settings.OPENAI_API_KEY,
    )
    chat_messages: List[MessageSchema] = conversation.messages
    chat_history = get_chat_history(chat_messages)
    logger.debug("Chat history: %s", chat_history)

    if conversation.documents:
        doc_titles = "\n".join(
            "- " + build_title_for_document(doc) for doc in conversation.documents
        )
    else:
        doc_titles = "No documents selected."

    curr_date = datetime.utcnow().strftime("%Y-%m-%d")
    chat_engine = OpenAIAgent.from_tools(
        # tools=top_level_sub_tools,
        tools=vector_query_engine_tools,
        llm=chat_llm,
        chat_history=chat_history,
        verbose=settings.VERBOSE,
        # system_prompt=SYSTEM_MESSAGE.format(doc_titles=doc_titles, curr_date=curr_date),
        callback_manager=service_context.callback_manager,
        max_function_calls=3,
    )

    return chat_engine
