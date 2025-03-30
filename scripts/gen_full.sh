# Default values
mix=false
sample=100  # Default sample size if not specified

# Function for displaying usage information
usage() {
    echo "Usage: $0 -i <word_list> [-m true|false] [-s <sample_size>]"
    echo "  -i: Input word list file (required)"
    echo "  -m: Enable mixing mode (true/false, default: false)"
    echo "  -s: Sample size (default: 100)"
    exit 1
}

# Process command line arguments
while getopts 'i:m:s:h' flag; do
    case "${flag}" in
        i) word_list=${OPTARG};;
        m) mix=${OPTARG};;
        s) sample=${OPTARG};;
        h) usage;;
        *) usage;;
    esac
done

# Validate required parameters
if [ -z "$word_list" ]; then
    echo "Error: Word list file (-i) is required."
    usage
fi

# Validate mix parameter
if [ "$mix" != "true" ] && [ "$mix" != "false" ]; then
    echo "Error: Mix parameter (-m) must be 'true' or 'false'."
    usage
fi

# Set mix flag for Python script
mix_flag=""
if [ "$mix" == "true" ]; then
    mix_flag="--mix"
fi

# Determine data directory based on word list type
if [[ "$word_list" == *"chinese"* ]]; then
    lang_type="chinese"
    if [ "$mix" == "true" ]; then
        data_dir="data/${lang_type}_anagram"
    else
        data_dir="data/${lang_type}"
    fi
elif [[ "$word_list" == *"english_words_simple"* ]]; then
    lang_type="english_simple"
    if [ "$mix" == "true" ]; then
        echo "Error: Mix mode is not supported for simple word list."
        exit 1
    fi
    data_dir="data/${lang_type}"
elif [[ "$word_list" == *"commonsenseqa_word"* ]]; then
    lang_type="commonsenseqa"
    if [ "$mix" == "true" ]; then
        echo "Error: Mix mode is not supported for simple word list."
        exit 1
    fi
    data_dir="data/${lang_type}"
elif [[ "$word_list" == *"simpleqa"* ]]; then
    lang_type="simpleqa"
    if [ "$mix" == "true" ]; then
        echo "Error: Mix mode is not supported for simple qa list."
        exit 1
    fi
    data_dir="data/${lang_type}"
else
    lang_type="english"
    if [ "$mix" == "true" ]; then
        data_dir="data/${lang_type}_anagram"
    else
        data_dir="data/${lang_type}"
    fi
fi

# Create data directory if it doesn't exist
mkdir -p "$data_dir"

# Log configuration details
echo "Configuration:"
echo "- Word list: $word_list"
echo "- Language type: $lang_type"
echo "- Mix mode: $mix"
echo "- Sample size: $sample"
echo "- Output directory: $data_dir"

# Execute the Python script
python gen_crossword.py \
    -i "$word_list" \
    -f np \
    -o "$data_dir" \
    --gen-full \
    --sample "$sample" \
    --no-replacement \
    $mix_flag

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Crossword generation completed successfully."
else
    echo "Error: Crossword generation failed with exit code $?."
    exit 1
fi