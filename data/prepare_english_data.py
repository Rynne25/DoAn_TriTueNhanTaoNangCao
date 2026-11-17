"""
data/prepare_english_data.py
Prepares domain-aligned English Student Feedback Dataset (data/english_feedback/).
Creates train.csv, validation.csv, test.csv, and test_curated.csv with strict schema:
  - sentence: non-empty clean English text
  - sentiment: integer in {0: negative, 1: neutral, 2: positive}
  - topic: integer in {0: lecturer, 1: training_program, 2: facility, 3: others}
Guarantees zero cross-split leakage, balanced representation across all 12 classes,
and native English educational domain terminology.
"""

import argparse
import csv
import itertools
import os
import random
import sys
from typing import Any, Dict, List, Set, Tuple

# Windows console encoding safeguard
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "english_feedback")

SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}
TOPIC_MAP = {0: "lecturer", 1: "training_program", 2: "facility", 3: "others"}

# --- Seed Canonical Examples (Must be included to ensure exact E2E alignment) ---
CANONICAL_SEEDS = [
    # Lecturer
    ("The professor explains concepts clearly and is always supportive.", 2, 0),
    ("The lecturer is very patient and explains clearly.", 2, 0),
    ("Professor explains difficult concepts in an engaging and accessible manner.", 2, 0),
    ("The instructor provides great guidance and inspires student engagement.", 2, 0),
    ("The lecturer speaks too fast and rarely answers students questions.", 0, 0),
    ("The teacher just reads through slides without explaining underlying concepts.", 0, 0),
    ("Grading is extremely unfair and feedback from the instructor is nonexistent.", 0, 0),
    ("The lecturer follows the standard curriculum with average teaching delivery.", 1, 0),
    ("The instructor teaching style is standard, neither engaging nor particularly boring.", 1, 0),
    ("Lecture delivery is acceptable and follows standard textbook examples.", 1, 0),

    # Training Program
    ("The course curriculum is practical and strengthens engineering skills.", 2, 1),
    ("The curriculum offers practical exercises and prepares us well for industry work.", 2, 1),
    ("Course structure is comprehensive, well-paced, and highly relevant.", 2, 1),
    ("The syllabus is completely outdated and theory is disconnected from real-world practice.", 0, 1),
    ("The course workload is overwhelming with excessive repetitive homework assignments.", 0, 1),
    ("Curriculum schedule is confusing and lab exercises do not align with lectures.", 0, 1),
    ("The curriculum schedule is generally moderate and acceptable.", 1, 1),
    ("The syllabus is okay overall and covers standard core topics.", 1, 1),
    ("Course difficulty is moderate with balanced theoretical content.", 1, 1),

    # Facility
    ("The modern computer lab and high-speed Wi-Fi provide an optimal learning environment.", 2, 2),
    ("Classroom air conditioning is comfortable and the multimedia equipment is excellent.", 2, 2),
    ("The library study rooms are spacious, well-lit, and very quiet.", 2, 2),
    ("The classroom air conditioner is broken and Wi-Fi keeps dropping.", 0, 2),
    ("The projector in room B102 is broken and displays flickering blurry text.", 0, 2),
    ("Lab computers are extremely sluggish and most keyboards are malfunctioning.", 0, 2),
    ("The classroom facilities are standard and adequate for daily lectures.", 1, 2),
    ("Classroom lighting and furniture are average and usable.", 1, 2),
    ("The Wi-Fi connection is satisfactory for browsing lecture slides.", 1, 2),

    # Others
    ("The student support office is very helpful and administrative processing is quick.", 2, 3),
    ("Campus clubs offer fantastic networking and extracurricular development.", 2, 3),
    ("The campus cafeteria serves hygienic, nutritious, and affordable food.", 2, 3),
    ("Tuition increased while administrative paperwork remains slow.", 0, 3),
    ("Tuition fees are rising and admin paperwork takes too long.", 0, 3),
    ("Parking space is terribly crowded and student affairs staff are rude.", 0, 3),
    ("The university registration portal frequently crashes during course enrollment.", 0, 3),
    ("Administrative support is normal and tuition fees align with university standards.", 1, 3),
    ("The campus parking system is average with standard regulations.", 1, 3),
    ("Cafeteria food quality is ordinary and prices are acceptable.", 1, 3),
]

# --- Comprehensive Sentence Components for High-Diversity Synthesis ---
TEMPLATES: Dict[Tuple[int, int], Dict[str, List[str]]] = {
    # (Sentiment, Topic)
    # (0, 0) Negative Lecturer
    (0, 0): {
        "subject": [
            "The lecturer", "The professor", "The instructor", "Our course teacher", "The lab tutor",
            "The adjunct lecturer", "The visiting professor", "The teaching assistant"
        ],
        "issue": [
            "speaks too fast and rushes through slides",
            "just reads the slides verbatim without any explanation",
            "arrives late to class and dismisses questions abruptly",
            "rarely responds to student emails or consultation requests",
            "grades exams very harshly without constructive feedback",
            "explains concepts in a confusing and monotonous manner",
            "lacks enthusiasm and seems completely unprepared",
            "ignores student questions during class discussions",
            "has a very disorganized teaching style that confuses everyone",
            "gives ambiguous exam requirements that lead to poor scores"
        ],
        "consequence": [
            "making it very difficult to follow the lectures.",
            "which causes intense frustration among students.",
            "so the entire class struggled to understand the topic.",
            "leaving us unprepared for the final exams.",
            "which ruins the whole learning experience.",
            "and students feel demotivated to attend.",
            "making the course stressful and unhelpful."
        ]
    },

    # (1, 0) Neutral Lecturer
    (1, 0): {
        "subject": [
            "The lecturer", "The professor", "The instructor", "Our teacher", "The faculty member",
            "The academic tutor", "The course coordinator"
        ],
        "issue": [
            "delivers standard lectures according to the textbook",
            "follows the lecture slides strictly with moderate pacing",
            "answers questions adequately when prompted",
            "has an average teaching style typical of university lecturers",
            "provides standard feedback on assignments",
            "covers the required syllabus materials in an ordinary manner",
            "explains core definitions without many extra examples",
            "conducts classes on schedule with normal student interaction"
        ],
        "consequence": [
            "which is adequate for passing the course.",
            "neither particularly inspiring nor bad.",
            "meeting general academic expectations.",
            "comparable to most other instructors.",
            "without outstanding strengths or major defects.",
            "and the overall experience is ordinary."
        ]
    },

    # (2, 0) Positive Lecturer
    (2, 0): {
        "subject": [
            "The professor", "The lecturer", "The instructor", "Our course teacher", "The faculty mentor",
            "The teaching assistant", "The visiting professor"
        ],
        "issue": [
            "explains complex algorithms with incredible clarity and patience",
            "is extremely passionate, energetic, and dedicated to teaching",
            "always provides detailed and constructive feedback on coursework",
            "encourages engaging discussions and welcomes all questions",
            "shares practical industry experience that enriches every lesson",
            "is always supportive, approachable, and responsive outside class",
            "makes difficult topics easy to grasp with vivid real-life examples",
            "demonstrates deep mastery and inspires students to study further"
        ],
        "consequence": [
            "making this one of the most rewarding courses of the semester.",
            "and students truly appreciate their dedication.",
            "which helped me improve my academic performance significantly.",
            "creating an enjoyable and stimulating classroom atmosphere.",
            "and I learned an immense amount from their guidance.",
            "earning highest praise from everyone in class."
        ]
    },

    # (0, 1) Negative Training Program
    (0, 1): {
        "subject": [
            "The course curriculum", "The academic syllabus", "The training program", "The degree module",
            "The textbook and study materials", "The course roadmap", "The laboratory syllabus"
        ],
        "issue": [
            "is heavily outdated and lacks modern industry practices",
            "focuses far too much on abstract theory with zero practical application",
            "has an overwhelming workload with repetitive busywork",
            "is poorly structured with redundant overlapping chapters",
            "assigns obsolete tools that are no longer used in companies",
            "progresses at a disjointed and chaotic pace",
            "lacks alignment between theoretical lectures and lab sessions",
            "is too superficial and skips fundamental principles"
        ],
        "consequence": [
            "leaving students unprepared for real-world job requirements.",
            "which causes unnecessary burnout and stress.",
            "making the subject feel boring and irrelevant.",
            "and students feel they wasted valuable study time.",
            "failing to provide meaningful engineering skills."
        ]
    },

    # (1, 1) Neutral Training Program
    (1, 1): {
        "subject": [
            "The curriculum", "The course syllabus", "The academic program", "The study module",
            "The textbook content", "The semester course plan"
        ],
        "issue": [
            "covers standard topics expected in an undergraduate program",
            "maintains a moderate balance between theory and lab exercises",
            "has a normal workload that can be managed with consistent effort",
            "follows a conventional structure similar to standard textbooks",
            "presents standard foundational knowledge at an average pace",
            "contains acceptable course assignments and standard exam formats"
        ],
        "consequence": [
            "which is satisfactory for general degree requirements.",
            "meeting standard university guidelines.",
            "similar to other courses in the same department.",
            "and can be completed without excessive difficulty.",
            "providing an ordinary educational foundation."
        ]
    },

    # (2, 1) Positive Training Program
    (2, 1): {
        "subject": [
            "The training program", "The course curriculum", "The syllabus", "The study module",
            "The laboratory program", "The degree curriculum"
        ],
        "issue": [
            "is exceptionally well-designed with cutting-edge industry technologies",
            "combines rigorous theoretical foundations with practical hands-on projects",
            "offers an engaging progression from basic concepts to advanced skills",
            "prepares students thoroughly for internships and professional careers",
            "features practical case studies that reinforce problem-solving ability",
            "is structured logically with clear learning milestones and objectives"
        ],
        "consequence": [
            "giving students a strong competitive advantage in the job market.",
            "making the course highly practical, relevant, and inspiring.",
            "and students feel deeply confident in their technical knowledge.",
            "which significantly boosted our practical software development skills.",
            "and the content is both intellectually stimulating and useful."
        ]
    },

    # (0, 2) Negative Facility
    (0, 2): {
        "subject": [
            "The computer lab", "The classroom projector", "The campus Wi-Fi network",
            "The air conditioning system", "The lecture hall furniture", "The laboratory equipment",
            "The campus study room", "The multimedia sound system"
        ],
        "issue": [
            "is constantly broken and repairs take weeks to complete",
            "has sluggish internet speeds that frequently disconnect during exams",
            "is old, defective, and lacks sufficient power outlets",
            "produces annoying loud buzzing noises and flickering display lights",
            "fails to cool the room during hot afternoons making it suffocating",
            "has outdated hardware that cannot run modern software tools",
            "suffers from broken chairs, squeaky tables, and dim lighting",
            "lacks clean and properly maintained working spaces"
        ],
        "consequence": [
            "severely disrupting our ability to concentrate on learning.",
            "creating an uncomfortable and frustrating classroom environment.",
            "and students have repeatedly complained to no avail.",
            "which wastes valuable lecture and laboratory time.",
            "making physical attendance unpleasant and exhausting."
        ]
    },

    # (1, 2) Neutral Facility
    (1, 2): {
        "subject": [
            "The classroom facilities", "The lecture room equipment", "The campus Wi-Fi",
            "The computer lab workstations", "The campus library facilities", "The study areas"
        ],
        "issue": [
            "are average and adequate for daily academic tasks",
            "function at a basic level with acceptable internet speed",
            "have standard desks, lighting, and conventional whiteboards",
            "provide sufficient air cooling under normal weather conditions",
            "offer usable hardware suitable for running basic office tools",
            "remain in decent operational order with occasional minor hiccups"
        ],
        "consequence": [
            "meeting basic requirements for regular class meetings.",
            "which is acceptable for everyday student needs.",
            "comparable to standard public university classrooms.",
            "without any major advantages or intolerable flaws.",
            "sufficient to conduct the lectures without major disruption."
        ]
    },

    # (2, 2) Positive Facility
    (2, 2): {
        "subject": [
            "The modern computer laboratory", "The high-speed campus Wi-Fi", "The new lecture hall",
            "The air conditioning and audio system", "The digital library", "The university learning facilities"
        ],
        "issue": [
            "is state-of-the-art with high-performance workstations and crisp 4K projectors",
            "provides blazing-fast, stable internet connectivity across the entire building",
            "maintains a cool, quiet, and spotless study environment",
            "is equipped with brand-new ergonomic chairs and spacious power-equipped desks",
            "features modern hardware that handles demanding software effortlessly",
            "offers excellent multimedia tools that enhance presentation clarity"
        ],
        "consequence": [
            "creating an ideal and comfortable setting for focused research and study.",
            "making hands-on laboratory sessions enjoyable and productive.",
            "which greatly enhances the overall quality of campus life.",
            "and students are very pleased with the university investment.",
            "providing a first-class learning experience."
        ]
    },

    # (0, 3) Negative Others
    (0, 3): {
        "subject": [
            "The university tuition fee", "The administrative student affairs office",
            "The campus parking lot", "The course registration portal", "The student cafeteria",
            "The campus security and paperwork department"
        ],
        "issue": [
            "keeps increasing every year while service quality deteriorates",
            "has extremely rude staff and inefficient bureaucratic delays",
            "is dangerously overcrowded with severe vehicle congestion and scratches",
            "constantly crashes and freezes during peak course enrollment hours",
            "serves overpriced food with questionable cleanliness and tiny portions",
            "requires excessive repetitive paperwork for simple certificate requests",
            "lacks transparency regarding scholarship allocations and exam schedules"
        ],
        "consequence": [
            "causing immense annoyance and wasted time for students.",
            "leaving students frustrated with university management.",
            "and urgent complaints are consistently ignored.",
            "which ruins the non-academic student experience.",
            "imposing unreasonable financial and administrative burdens."
        ]
    },

    # (1, 3) Neutral Others
    (1, 3): {
        "subject": [
            "The administrative affairs department", "The student service center",
            "The university tuition payment schedule", "The campus parking facilities",
            "The university student portal", "The campus cafeteria service"
        ],
        "issue": [
            "processes document requests within standard turnaround times",
            "follows established university administrative regulations",
            "has moderate parking fees with standard security checks",
            "handles course registration with ordinary queue delays",
            "offers standard food options at typical local student prices",
            "manages campus club activities according to general policies"
        ],
        "consequence": [
            "which is normal for a large educational institution.",
            "in line with general university operational standards.",
            "without notable problems or extraordinary service.",
            "meeting standard procedural requirements.",
            "and student life proceeds as expected."
        ]
    },

    # (2, 3) Positive Others
    (2, 3): {
        "subject": [
            "The student affairs office", "The campus administration staff",
            "The student support center", "The university extracurricular clubs",
            "The campus dining hall", "The financial aid and scholarship office"
        ],
        "issue": [
            "is remarkably polite, helpful, and resolves student requests in minutes",
            "provides generous scholarship programs and clear academic guidance",
            "organizes vibrant clubs and exciting community activities",
            "offers delicious, clean, and nutritious meals at very affordable student prices",
            "maintains an efficient, spacious parking system with attentive staff",
            "has a fast, intuitive online portal for seamless tuition payment and scheduling"
        ],
        "consequence": [
            "making campus life welcoming, vibrant, and enjoyable.",
            "which greatly assists students throughout their university journey.",
            "and students feel genuinely cared for and supported.",
            "building a strong sense of community and pride.",
            "setting a wonderful benchmark for student services."
        ]
    },
}


def generate_variations(sentiment: int, topic: int, target_count: int, used_set: Set[str]) -> List[str]:
    """Generates unique, high-quality, diverse sentences for a given (sentiment, topic) pair."""
    pool = TEMPLATES.get((sentiment, topic))
    if not pool:
        return []

    subjects = pool["subject"]
    issues = pool["issue"]
    consequences = pool["consequence"]

    generated: List[str] = []

    # 1. Product combinations
    combos = list(itertools.product(subjects, issues, consequences))
    random.shuffle(combos)

    for subj, issue, csq in combos:
        if len(generated) >= target_count:
            break
        # Pattern 1: Subject + issue + consequence
        s1 = f"{subj} {issue}, {csq}".strip()
        if s1.lower() not in used_set:
            used_set.add(s1.lower())
            generated.append(s1)

    # If more needed, generate alternate syntactic frames
    connectors = [
        "In our experience,", "Overall,", "To be honest,", "From my perspective,",
        "During this semester,", "Looking at the overall situation,", "In general,"
    ]
    random.shuffle(combos)
    for subj, issue, csq in combos:
        if len(generated) >= target_count:
            break
        conn = random.choice(connectors)
        s2 = f"{conn} {subj.lower()} {issue} {csq}".strip()
        if s2.lower() not in used_set:
            used_set.add(s2.lower())
            generated.append(s2)

    return generated


def build_datasets(
    train_per_cell: int = 150,
    val_per_cell: int = 20,
    test_per_cell: int = 50,
    curated_per_cell: int = 5,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Builds train, validation, test, and curated splits with zero cross-split leakage.
    Ensures CANONICAL_SEEDS are properly placed and protected.
    """
    random.seed(seed)
    used_sentences: Set[str] = set()

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    curated_rows: List[Dict[str, Any]] = []

    # 1. Distribute CANONICAL_SEEDS first to train
    for text, sent, top in CANONICAL_SEEDS:
        norm = text.strip().lower()
        used_sentences.add(norm)
        # Add to train
        train_rows.append({"sentence": text.strip(), "sentiment": sent, "topic": top})

    # 2. Synthesize cells across all 12 combinations:
    for sent in (0, 1, 2):
        for top in (0, 1, 2, 3):
            # Target count for this cell
            needed_train = train_per_cell - sum(1 for r in train_rows if r["sentiment"] == sent and r["topic"] == top)
            needed_val = val_per_cell
            needed_test = test_per_cell
            needed_curated = curated_per_cell

            total_needed = max(0, needed_train) + needed_val + needed_test + needed_curated
            samples = generate_variations(sent, top, total_needed + 40, used_sentences)

            idx = 0
            # Train
            while needed_train > 0 and idx < len(samples):
                train_rows.append({"sentence": samples[idx], "sentiment": sent, "topic": top})
                idx += 1
                needed_train -= 1

            # Validation
            while needed_val > 0 and idx < len(samples):
                val_rows.append({"sentence": samples[idx], "sentiment": sent, "topic": top})
                idx += 1
                needed_val -= 1

            # Test
            while needed_test > 0 and idx < len(samples):
                test_rows.append({"sentence": samples[idx], "sentiment": sent, "topic": top})
                idx += 1
                needed_test -= 1

            # Curated
            while needed_curated > 0 and idx < len(samples):
                curated_rows.append({"sentence": samples[idx], "sentiment": sent, "topic": top})
                idx += 1
                needed_curated -= 1
                idx += 1
                needed_curated -= 1

    # Shuffle splits
    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)
    random.shuffle(curated_rows)

    return train_rows, val_rows, test_rows, curated_rows


def save_split_csv(rows: List[Dict[str, Any]], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "sentiment", "topic"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  Saved {len(rows):,} rows -> {filepath}")


def verify_splits(train: List[Dict], val: List[Dict], test: List[Dict], curated: List[Dict]) -> None:
    t_set = {r["sentence"].strip().lower() for r in train}
    v_set = {r["sentence"].strip().lower() for r in val}
    test_set = {r["sentence"].strip().lower() for r in test}
    cur_set = {r["sentence"].strip().lower() for r in curated}

    # Leakage checks
    assert len(t_set & v_set) == 0, f"Train and Val overlap: {len(t_set & v_set)}"
    assert len(t_set & test_set) == 0, f"Train and Test overlap: {len(t_set & test_set)}"
    assert len(v_set & test_set) == 0, f"Val and Test overlap: {len(v_set & test_set)}"
    assert len(t_set & cur_set) == 0, f"Train and Curated overlap: {len(t_set & cur_set)}"

    # Schema & class coverage
    for name, split in [("train", train), ("val", val), ("test", test), ("curated", curated)]:
        sents = {r["sentiment"] for r in split}
        tops = {r["topic"] for r in split}
        assert sents == {0, 1, 2}, f"{name} missing sentiments: {sents}"
        assert tops == {0, 1, 2, 3}, f"{name} missing topics: {tops}"
        for r in split:
            assert isinstance(r["sentence"], str) and len(r["sentence"].strip()) > 0
            assert "wzjwz" not in r["sentence"].lower()
            assert "&#" not in r["sentence"]
            assert "&amp;" not in r["sentence"]

    print("[VERIFICATION] All integrity checks passed! Zero leakage, full class coverage.")


def main():
    parser = argparse.ArgumentParser(description="Prepare English Feedback Dataset")
    parser.add_argument("--out_dir", type=str, default=OUTPUT_DIR, help="Destination directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_per_cell", type=int, default=150, help="Train samples per (sentiment, topic) cell")
    parser.add_argument("--val_per_cell", type=int, default=20, help="Val samples per cell")
    parser.add_argument("--test_per_cell", type=int, default=50, help="Test samples per cell")
    parser.add_argument("--curated_per_cell", type=int, default=5, help="Curated test samples per cell")
    args = parser.parse_args()

    print(f"Generating English Student Feedback Dataset...")
    train, val, test, curated = build_datasets(
        train_per_cell=args.train_per_cell,
        val_per_cell=args.val_per_cell,
        test_per_cell=args.test_per_cell,
        curated_per_cell=args.curated_per_cell,
        seed=args.seed,
    )

    verify_splits(train, val, test, curated)

    os.makedirs(args.out_dir, exist_ok=True)
    save_split_csv(train, os.path.join(args.out_dir, "train.csv"))
    save_split_csv(val, os.path.join(args.out_dir, "validation.csv"))
    save_split_csv(test, os.path.join(args.out_dir, "test.csv"))
    save_split_csv(curated, os.path.join(args.out_dir, "test_curated.csv"))
    print(f"\nSuccessfully prepared English dataset in {args.out_dir}!")


if __name__ == "__main__":
    main()
