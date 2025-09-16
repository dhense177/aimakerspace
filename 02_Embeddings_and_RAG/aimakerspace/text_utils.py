import os
from typing import List


class PlainTextLoader:
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def can_handle(self, path: str) -> bool:
        return os.path.isfile(path) and path.endswith(".txt")

    def load(self, path: str) -> List[str]:
        with open(path, "r", encoding=self.encoding) as f:
            return [f.read()]


class PdfFileLoader:
    def can_handle(self, path: str) -> bool:
        return os.path.isfile(path) and path.endswith(".pdf")

    def load(self, path: str) -> List[str]:
        # Try TOC-based extraction with PyMuPDF (fitz) first
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(path)
            toc = doc.get_toc(simple=True)
            if toc:
                sections: List[str] = []
                for i, (_, title, start1) in enumerate(toc):
                    start0 = max(start1 - 1, 0)
                    end0 = (toc[i + 1][2] - 2) if i + 1 < len(toc) else doc.page_count - 1
                    text_parts: List[str] = []
                    for pno in range(start0, end0 + 1):
                        # get_text("text") is the default; could use blocks/dict for layout-aware
                        text_parts.append(doc[pno].get_text())
                    section_text = f"{title}\n\n" + "\n".join(text_parts)
                    sections.append(section_text)
                if sections:
                    return sections
            # If no TOC, fall back to full-document extraction via PyMuPDF
            full_text = []
            for pno in range(doc.page_count):
                full_text.append(doc[pno].get_text())
            return ["\n".join(full_text)]
        except Exception:
            # Fallback to pypdf if PyMuPDF not available or fails
            from pypdf import PdfReader
            reader = PdfReader(path)
            texts = []
            for page in reader.pages:
                texts.append(page.extract_text() or "")
            return ["\n".join(texts)]


class DirectoryLoader:
    def __init__(self, loaders: List[object]):
        self.loaders = loaders

    def load(self, directory: str) -> List[str]:
        documents = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                for loader in self.loaders:
                    if loader.can_handle(file_path):
                        documents.extend(loader.load(file_path))
                        break
        return documents


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding
        self._loaders = [PlainTextLoader(encoding=self.encoding), PdfFileLoader()]

    def load(self):
        if os.path.isdir(self.path):
            directory_loader = DirectoryLoader(self._loaders)
            self.documents = directory_loader.load(self.path)
        elif os.path.isfile(self.path):
            for loader in self._loaders:
                if loader.can_handle(self.path):
                    self.documents = loader.load(self.path)
                    break
            else:
                raise ValueError("Unsupported file type. Supported: .txt, .pdf")
        else:
            raise ValueError("Provided path is neither a valid directory nor a file.")

    def load_documents(self):
        self.load()
        return self.documents

class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


class WordTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000
    ):
        self.chunk_size = chunk_size

    def split(self, text: str) -> List[str]:
        chunks = []

        if len(text)<self.chunk_size:
            return [text]

        chunks = [text]

        while all(len(s) < self.chunk_size for s in chunks)==False:
            texts_to_shorten = [i for i in chunks if len(i)>self.chunk_size]
            chunks = [x for x in chunks if x not in texts_to_shorten]
            for t in texts_to_shorten:
                # First try splitting on periods
                if t.find(".")>0:
                    t_list = t.split(".")
                    first_half = '. '.join([i.replace("\n","") for i in t_list[:len(t_list)//2]]).strip()
                    second_half = '. '.join([i.replace("\n","") for i in t_list[len(t_list)//2:]]).strip()
                # Then try splitting on words
                else:
                    t_list = t.split()
                    first_half = ' '.join([i.replace("\n","") for i in t_list[:len(t_list)//2]]).strip()
                    second_half = ' '.join([i.replace("\n","") for i in t_list[len(t_list)//2:]]).strip()
                chunks.append(first_half)
                chunks.append(second_half)

        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
