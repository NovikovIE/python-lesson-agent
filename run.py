import sys
from src.main_graph import generate_lesson

def main():
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = "Создай урок про dataclasses в Python"

    print(f"[run.py] generation for: '{topic}'")
    
    final_lesson = generate_lesson(topic)
    
    output_filename = f"урок_{topic.replace(' ', '_')[:30]}.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_lesson)
        
    print(f"[run.py] generated lesson to '{output_filename}'")
    print("\n[run.py] content:")
    print(final_lesson)

if __name__ == "__main__":
    main()