from pydantic import BaseModel, Field


class DocumentIdTitle(BaseModel):
    """
    This is a document ID and title pair. It is meant allow the retrieval of a document from a Chroma database.
    """

    doc_id: str = Field(description="The ID of the document to retrieve")
    doc_title: str = Field(description="The title of the document to retrieve")


class DocumentsQuery(BaseModel):
    """
    This is a query to the context retrieval agent. It is used to retrieve a list of documents from a Chroma database.
    """

    documents: list[DocumentIdTitle] = Field(
        description="The list of documents to retrieve, as a list of DocumentIdTitle objects"
    )
