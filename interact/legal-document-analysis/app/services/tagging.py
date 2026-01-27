try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback for older langchain versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter

def classify_text(text: str):
    """Classify text line as heading or body."""
    return "heading" if text.isupper() else "body"

def tag_documents(data: str) -> list:
    """Tag extracted text into structured sections with headings and documents."""
    tagged_documents, current_heading, current_paragraph = [], None, ""

    for line in data.split("\n"):
        line = line.strip()
        if not line:
            continue

        if classify_text(line) == "heading":
            if current_heading:
                tagged_documents.append(
                    {"heading": current_heading, "body": current_paragraph.strip()}
                )
            current_heading, current_paragraph = line, ""
        else:
            current_paragraph += " " + line

    if current_heading:
        tagged_documents.append(
            {"heading": current_heading, "body": current_paragraph.strip()}
        )

    # Split body into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=30,
    )

    for section in tagged_documents:
        documents = splitter.create_documents([section["body"]])
        section["documents"] = [doc.page_content for doc in documents]

    return tagged_documents
