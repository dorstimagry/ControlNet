# PowerPoint Presentation Generator

This script converts the markdown presentation content (`presentation_content.md`) into a PowerPoint (.pptx) file.

## Installation

First, install the required dependency:

```bash
pip install python-pptx
```

## Usage

### Basic Usage

```bash
python scripts/create_presentation.py
```

This will:
- Read from `docs/presentation_content.md`
- Create `docs/presentation.pptx`

### Custom Input/Output

```bash
python scripts/create_presentation.py \
    --input docs/presentation_content.md \
    --output docs/my_presentation.pptx
```

## Features

The script automatically:

- ✅ Converts markdown sections to PowerPoint slides
- ✅ Handles bullet points and numbered lists
- ✅ Formats code blocks as monospace text
- ✅ **Renders LaTeX equations as images** (using matplotlib)
- ✅ **Finds and inserts actual plot images** from evaluation results
- ✅ Extracts figure placeholders and searches for matching images
- ✅ Creates section header slides for major sections
- ✅ Splits long content across multiple slides
- ✅ Handles tables, blockquotes, and other markdown elements

### Image and Formula Support

- **Equations**: LaTeX formulas (e.g., `\[ J(\pi) = ... \]`) are automatically rendered as images and inserted into slides
- **Figures**: The script searches for images matching figure descriptions in:
  - `evaluation/results/plots/`
  - `evaluation/results/plots/closed_loop_cpp_comparison_with_initial_error/`
  - Any additional directories specified with `--figure-dirs`

## Limitations

- **Mermaid diagrams**: Currently skipped (you'll need to manually add diagrams or convert them to images)
- **LaTeX formulas**: Shown as text placeholders (you may want to convert to images)
- **Complex formatting**: Some advanced markdown features may not render perfectly

## Post-Processing Recommendations

After generating the PowerPoint:

1. **Review inserted images**: Check that figures and equations were inserted correctly
2. **Add missing diagrams**: Convert Mermaid diagrams to images and insert them manually
3. **Verify formulas**: Check that LaTeX formulas rendered correctly (complex formulas may need manual adjustment)
4. **Add missing figures**: If some figures weren't found automatically, insert them manually based on slide notes
5. **Review formatting**: Check bullet points, spacing, and text formatting
6. **Add transitions**: Add slide transitions and animations if desired

### Custom Figure Directories

If you have figures in other locations, specify them:

```bash
python scripts/create_presentation.py \
    --input docs/presentation_content.md \
    --output docs/presentation.pptx \
    --figure-dirs path/to/figures1 path/to/figures2
```

## Troubleshooting

### Import Error

If you get `ModuleNotFoundError: No module named 'pptx'`:

```bash
pip install python-pptx
```

### File Not Found

Make sure `docs/presentation_content.md` exists, or specify the correct path with `--input`.

### Slides Look Wrong

The script does its best to parse markdown, but complex formatting may need manual adjustment in PowerPoint.

