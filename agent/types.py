from pydantic import BaseModel, Field


class DocumentIdTitle(BaseModel):
    """
    This is a document ID and title pair. It is meant allow the retrieval of a document from a Chroma database.
    """

    doc_id: str = Field(
        description="The ID of the document to retrieve, typically an md5 hash, should not include the document title"
    )
    doc_title: str = Field(description="The title of the document to retrieve")

    def __str__(self):
        return f"{{Document Title: {self.doc_title}\nID: {self.doc_id}}}"


class DocumentsQuery(BaseModel):
    """
    This is a query from the context retrieval agent. It is used to retrieve a list of documents from a Chroma database.
    """

    documents: list[DocumentIdTitle] = Field(
        description="The list of documents to retrieve, as a list of DocumentIdTitle objects"
    )


class WikiPage(BaseModel):
    title: str = Field(description="The title of the page")
    url: str = Field(description="The URL of the page")
    raw: str = Field(description="The raw content of the page")
