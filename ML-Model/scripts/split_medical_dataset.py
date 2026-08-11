from pathlib import Path
import random
import shutil
import hashlib

import _bootstrap  # noqa: F401


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw source folder containing all domain subdirectories before splitting.
SOURCE_DIR = PROJECT_ROOT / "data" / "raw"

# Output folder for generated train / validation / test splits.
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
}


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

def validate_configuration():

    total_ratio = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(
            "Train + Validation + Test ratios "
            "must equal 1.0"
        )

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"\nSource directory does not exist:\n"
            f"{SOURCE_DIR}"
        )

    if not SOURCE_DIR.is_dir():
        raise NotADirectoryError(
            f"\nSource path is not a directory:\n"
            f"{SOURCE_DIR}"
        )

    if OUTPUT_DIR.resolve() == SOURCE_DIR.resolve():
        raise ValueError(
            "SOURCE_DIR and OUTPUT_DIR cannot be the same."
        )


# ============================================================
# FIND IMAGES
# ============================================================

def get_images(folder: Path) -> list[Path]:
    """
    Find all supported image files recursively.
    """

    return sorted(
        file
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower()
        in IMAGE_EXTENSIONS
    )


# ============================================================
# FILE HASH
# ============================================================

def file_hash(file_path: Path) -> str:
    """
    Calculate SHA-256 hash.

    Used to detect exact duplicate images.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# FIND DOMAINS
# ============================================================

def find_domains() -> list[Path]:

    domains = sorted(
        folder
        for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    )

    if not domains:
        raise RuntimeError(
            f"\nNo domain folders found in:\n"
            f"{SOURCE_DIR}"
        )

    return domains


# ============================================================
# FIND CLASSES
# ============================================================

def find_classes(domain_dir: Path) -> list[Path]:

    classes = sorted(
        folder
        for folder in domain_dir.iterdir()
        if folder.is_dir()
    )

    if not classes:
        raise RuntimeError(
            f"\nNo class folders found in:\n"
            f"{domain_dir}"
        )

    return classes


# ============================================================
# SPLIT IMAGES
# ============================================================

def split_images(
    images: list[Path],
    class_name: str,
    domain_name: str,
):
    """
    Split one domain/class independently.

    80% Train
    10% Validation
    10% Test
    """

    images = images.copy()

    # Create deterministic seed for every class.
    class_seed = (
        RANDOM_SEED
        + sum(
            ord(c)
            for c in domain_name
        )
        + sum(
            ord(c)
            for c in class_name
        )
    )

    rng = random.Random(
        class_seed
    )

    rng.shuffle(images)

    total = len(images)

    train_count = int(
        total * TRAIN_RATIO
    )

    validation_count = int(
        total * VALIDATION_RATIO
    )

    train = images[
        :train_count
    ]

    validation = images[
        train_count:
        train_count + validation_count
    ]

    test = images[
        train_count + validation_count:
    ]

    return (
        train,
        validation,
        test,
    )


# ============================================================
# CREATE DESTINATION
# ============================================================

def create_destination(
    split: str,
    domain_name: str,
    class_name: str,
) -> Path:

    destination = (
        OUTPUT_DIR
        / split
        / domain_name
        / class_name
    )

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    return destination


# ============================================================
# COPY IMAGES
# ============================================================

def copy_images(
    images: list[Path],
    split: str,
    domain_name: str,
    class_name: str,
):

    destination = create_destination(
        split,
        domain_name,
        class_name,
    )

    for image in images:

        target = (
            destination
            / image.name
        )

        if target.exists():
            raise FileExistsError(
                f"\nDestination file already exists:\n"
                f"{target}"
            )

        shutil.copy2(
            image,
            target
        )


# ============================================================
# VERIFY SPLIT
# ============================================================

def verify_split(
    train_images: list[Path],
    validation_images: list[Path],
    test_images: list[Path],
    domain_name: str,
    class_name: str,
):
    """
    Verify that source images are assigned
    to only one split.
    """

    train_set = {
        image.resolve()
        for image in train_images
    }

    validation_set = {
        image.resolve()
        for image in validation_images
    }

    test_set = {
        image.resolve()
        for image in test_images
    }

    if train_set & validation_set:

        raise RuntimeError(
            f"\nTrain/Validation overlap detected:\n"
            f"{domain_name}/{class_name}"
        )

    if train_set & test_set:

        raise RuntimeError(
            f"\nTrain/Test overlap detected:\n"
            f"{domain_name}/{class_name}"
        )

    if validation_set & test_set:

        raise RuntimeError(
            f"\nValidation/Test overlap detected:\n"
            f"{domain_name}/{class_name}"
        )


# ============================================================
# VERIFY OUTPUT
# ============================================================

def count_output_images(
    split: str,
) -> int:

    directory = (
        OUTPUT_DIR / split
    )

    if not directory.exists():
        return 0

    return len([
        file
        for file in directory.rglob("*")
        if file.is_file()
        and file.suffix.lower()
        in IMAGE_EXTENSIONS
    ])


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("MULTI-DOMAIN MEDICAL DATASET SPLITTER")
    print("=" * 75)

    print("\nSOURCE:")
    print(SOURCE_DIR)

    print("\nOUTPUT:")
    print(OUTPUT_DIR)

    print("\nSPLIT:")
    print("Train      : 80%")
    print("Validation : 10%")
    print("Test       : 10%")

    print("\nRandom Seed:")
    print(RANDOM_SEED)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    validate_configuration()

    # --------------------------------------------------------
    # Prevent accidental overwrite
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():

        existing_files = [
            file
            for file in OUTPUT_DIR.rglob("*")
            if file.is_file()
        ]

        if existing_files:

            raise RuntimeError(
                f"\nOUTPUT DIRECTORY IS NOT EMPTY:\n"
                f"{OUTPUT_DIR}\n\n"
                "Delete or rename the old "
                "Medical_Split folder first."
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find domains
    # --------------------------------------------------------

    domains = find_domains()

    print(
        f"\nFound {len(domains)} domains:"
    )

    for domain in domains:

        print(
            f"  - {domain.name}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    grand_original = 0
    grand_train = 0
    grand_validation = 0
    grand_test = 0

    # --------------------------------------------------------
    # PROCESS DOMAINS
    # --------------------------------------------------------

    for domain_dir in domains:

        domain_name = domain_dir.name

        print("\n")
        print("=" * 75)
        print(
            f"DOMAIN: {domain_name}"
        )
        print("=" * 75)

        classes = find_classes(
            domain_dir
        )

        print(
            f"\nClasses found: "
            f"{len(classes)}"
        )

        # ----------------------------------------------------
        # PROCESS CLASSES
        # ----------------------------------------------------

        domain_original = 0
        domain_train = 0
        domain_validation = 0
        domain_test = 0

        for class_dir in classes:

            class_name = class_dir.name

            images = get_images(
                class_dir
            )

            if not images:

                print(
                    f"\nWARNING: "
                    f"No images found:"
                    f"\n{domain_name}/{class_name}"
                )

                continue

            # ------------------------------------------------
            # Detect exact duplicates
            # ------------------------------------------------

            unique_images = []
            hashes = {}
            duplicate_count = 0

            for image in images:

                image_hash = file_hash(
                    image
                )

                if image_hash in hashes:

                    duplicate_count += 1

                    print(
                        "\nWARNING: Exact duplicate:"
                    )

                    print(
                        f"  {image}"
                    )

                    continue

                hashes[
                    image_hash
                ] = image

                unique_images.append(
                    image
                )

            images = unique_images

            # ------------------------------------------------
            # Split
            # ------------------------------------------------

            (
                train,
                validation,
                test,
            ) = split_images(
                images,
                class_name,
                domain_name,
            )

            # ------------------------------------------------
            # Verify
            # ------------------------------------------------

            verify_split(
                train,
                validation,
                test,
                domain_name,
                class_name,
            )

            # ------------------------------------------------
            # Copy
            # ------------------------------------------------

            copy_images(
                train,
                "Train",
                domain_name,
                class_name,
            )

            copy_images(
                validation,
                "Validation",
                domain_name,
                class_name,
            )

            copy_images(
                test,
                "Test",
                domain_name,
                class_name,
            )

            # ------------------------------------------------
            # Counts
            # ------------------------------------------------

            original_count = len(
                images
            )

            train_count = len(
                train
            )

            validation_count = len(
                validation
            )

            test_count = len(
                test
            )

            domain_original += (
                original_count
            )

            domain_train += (
                train_count
            )

            domain_validation += (
                validation_count
            )

            domain_test += (
                test_count
            )

            print(
                f"\n{class_name}"
            )

            print(
                f"  Original    : "
                f"{original_count:,}"
            )

            print(
                f"  Train       : "
                f"{train_count:,}"
            )

            print(
                f"  Validation  : "
                f"{validation_count:,}"
            )

            print(
                f"  Test        : "
                f"{test_count:,}"
            )

            if duplicate_count:

                print(
                    f"  Duplicates removed: "
                    f"{duplicate_count}"
                )

        # ----------------------------------------------------
        # Domain summary
        # ----------------------------------------------------

        print("\n")
        print(
            f"{domain_name} SUMMARY"
        )

        print(
            f"  Original    : "
            f"{domain_original:,}"
        )

        print(
            f"  Train       : "
            f"{domain_train:,}"
        )

        print(
            f"  Validation  : "
            f"{domain_validation:,}"
        )

        print(
            f"  Test        : "
            f"{domain_test:,}"
        )

        grand_original += domain_original
        grand_train += domain_train
        grand_validation += domain_validation
        grand_test += domain_test

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    split_total = (
        grand_train
        + grand_validation
        + grand_test
    )

    print("\n")
    print("=" * 75)
    print("FINAL DATASET SUMMARY")
    print("=" * 75)

    print(
        f"\nOriginal unique images : "
        f"{grand_original:,}"
    )

    print(
        f"Train                  : "
        f"{grand_train:,}"
    )

    print(
        f"Validation             : "
        f"{grand_validation:,}"
    )

    print(
        f"Test                   : "
        f"{grand_test:,}"
    )

    print(
        f"Split total            : "
        f"{split_total:,}"
    )

    # --------------------------------------------------------
    # Verify totals
    # --------------------------------------------------------

    if split_total != grand_original:

        raise RuntimeError(
            "\nERROR: Split total does not "
            "match original total!"
        )

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    print("\n")
    print(
        "OUTPUT VERIFICATION"
    )

    print("-" * 50)

    train_output = count_output_images(
        "Train"
    )

    validation_output = count_output_images(
        "Validation"
    )

    test_output = count_output_images(
        "Test"
    )

    print(
        f"Train       : "
        f"{train_output:,}"
    )

    print(
        f"Validation  : "
        f"{validation_output:,}"
    )

    print(
        f"Test        : "
        f"{test_output:,}"
    )

    output_total = (
        train_output
        + validation_output
        + test_output
    )

    print(
        f"Total       : "
        f"{output_total:,}"
    )

    if output_total != grand_original:

        raise RuntimeError(
            "\nERROR: Output image count "
            "does not match source count!"
        )

    # ========================================================
    # SUCCESS
    # ========================================================

    print("\n")
    print("=" * 75)
    print("SUCCESS")
    print("=" * 75)

    print(
        "\nNew dataset created successfully:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nOriginal dataset was NOT modified."
    )

    print(
        "\nStructure:"
    )

    print(
        "Medical_Split/"
    )

    print(
        "├── Train/"
    )

    print(
        "│   ├── Eye disease/"
    )

    print(
        "│   ├── FYP skin disease Dataset/"
    )

    print(
        "│   └── Oral Cancer/"
    )

    print(
        "├── Validation/"
    )

    print(
        "│   ├── Eye disease/"
    )

    print(
        "│   ├── FYP skin disease Dataset/"
    )

    print(
        "│   └── Oral Cancer/"
    )

    print(
        "└── Test/"
    )

    print(
        "    ├── Eye disease/"
    )

    print(
        "    ├── FYP skin disease Dataset/"
    )

    print(
        "    └── Oral Cancer/"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()