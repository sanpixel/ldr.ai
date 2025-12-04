# AutoCAD LISP Commands for LDR.AI

These AutoCAD LISP commands help you work with legal description data in AutoCAD.

## Installation

1. Open AutoCAD
2. Type `APPLOAD` at the command line
3. Browse to and select the `.lsp` file you want to load
4. Click "Load"

## Available Commands

### DRAWBEARINGS
Interactively draw property lines from bearing data.

**Usage:**
1. Type `DRAWBEARINGS` at command line
2. Click to set Point of Beginning (POB)
3. Enter bearing (e.g., "N 45d 30m 15s E")
4. Enter distance in feet
5. Repeat for each line
6. Press ENTER when done

### IMPORTSURVEY
Import survey data from CSV file exported from ldr.ai.

**Usage:**
1. Export your survey data from ldr.ai as CSV
2. Type `IMPORTSURVEY` at command line
3. Select the CSV file
4. Click to set Point of Beginning (POB)
5. Lines will be drawn automatically

**CSV Format:**
```
Bearing,Distance,Monument
N 45d 30m 15s E,150.00,Iron Pin Found
S 73d 32m 01s W,125.50,Concrete Monument
```

### LABELMON
Add monument labels to survey points.

**Usage:**
1. Type `LABELMON` at command line
2. Click on monument location
3. Enter monument description
4. Repeat for each monument
5. Press ENTER when done

## Notes

- All distances are in feet
- Bearings use surveyor's notation (quadrant bearings)
- These files are in .gitignore and won't be pushed to the repository
- Customize the LISP files as needed for your workflow

## Customization

You can modify these files to:
- Change text sizes and styles
- Adjust marker sizes
- Add layers for different elements
- Customize dimension styles
- Add more automation features
