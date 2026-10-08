from ragforge.core.config import get_settings
from ragforge.app.generation.service import answer_query


settings = get_settings()

response = answer_query(
    settings,
    "What happens to the security deposit when the tenant vacates?",
)

print("\n=== ANSWER ===")
print(response.answer)

print("\n=== SOURCES ===")

for i, source in enumerate(response.sources, start=1):
    print(f"\n--- Source {i} ---")
    print(f"Score: {source.score:.4f}")
    print(f"Page: {source.metadata.get('page_number')}")
    print(f"Section: {source.metadata.get('section_number')}")
    print(source.text)