"""
Fingerprinting system for Legal Description Reader
Creates normalized representations for comparison of extracted line data
"""

import re
from typing import List, Set, Dict, Any
from utils.schema import LineData, SchemaOutput, LineType
import math


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent fingerprinting
    
    Args:
        text: Raw text to normalize
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    normalized = text.lower().strip()
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove common punctuation that doesn't affect meaning
    normalized = re.sub(r'[.,;:()"]', '', normalized)
    
    # Normalize directional terms
    directional_replacements = {
        'northerly': 'north',
        'southerly': 'south',
        'easterly': 'east',
        'westerly': 'west',
        'northeasterly': 'northeast',
        'northwesterly': 'northwest',
        'southeasterly': 'southeast',
        'southwesterly': 'southwest'
    }
    
    for old, new in directional_replacements.items():
        normalized = normalized.replace(old, new)
    
    return normalized


def round_numeric_value(value: float, precision: int = 2) -> float:
    """
    Round numeric values for consistent comparison
    
    Args:
        value: Numeric value to round
        precision: Number of decimal places
        
    Returns:
        float: Rounded value
    """
    if value is None:
        return 0.0
    
    return round(float(value), precision)


def generate_line_fingerprint(line: LineData) -> str:
    """
    Generate normalized fingerprint for a single line
    
    Args:
        line: LineData to fingerprint
        
    Returns:
        str: Normalized fingerprint string
    """
    if line.type == LineType.COURSE.value:
        return generate_course_fingerprint(line)
    elif line.type == LineType.CURVE.value:
        return generate_curve_fingerprint(line)
    elif line.type == LineType.REF_SEGMENT.value:
        return generate_ref_segment_fingerprint(line)
    elif line.type == LineType.REF_CURVE.value:
        return generate_ref_curve_fingerprint(line)
    else:
        return f"unknown|{line.type}|{normalize_text(line.raw)}"


def generate_course_fingerprint(line: LineData) -> str:
    """
    Generate fingerprint for course line
    Format: course|S|45|12|30|E|125.00
    
    Args:
        line: Course LineData
        
    Returns:
        str: Course fingerprint
    """
    # Extract and normalize components
    cardinal_ns = (line.cardinal_ns or '').upper()
    cardinal_ew = (line.cardinal_ew or '').upper()
    
    # Normalize cardinal directions to single letters
    if cardinal_ns.startswith('N'):
        ns = 'N'
    elif cardinal_ns.startswith('S'):
        ns = 'S'
    else:
        ns = 'N'  # Default
    
    if cardinal_ew.startswith('E'):
        ew = 'E'
    elif cardinal_ew.startswith('W'):
        ew = 'W'
    else:
        ew = 'E'  # Default
    
    # Normalize numeric values
    degrees = int(line.degrees or 0)
    minutes = int(line.minutes or 0)
    seconds = round_numeric_value(line.seconds or 0.0, 2)
    distance = round_numeric_value(line.distance or 0.0, 2)
    
    return f"course|{ns}|{degrees}|{minutes}|{seconds}|{ew}|{distance}"


def generate_curve_fingerprint(line: LineData) -> str:
    """
    Generate fingerprint for curve line
    Format: curve|S|45|12|30|E|125.00
    
    Args:
        line: Curve LineData
        
    Returns:
        str: Curve fingerprint
    """
    # Use same format as course for now
    # Could be extended with curve-specific parameters (radius, arc length, etc.)
    return generate_course_fingerprint(line).replace('course|', 'curve|')


def generate_ref_segment_fingerprint(line: LineData) -> str:
    """
    Generate fingerprint for reference segment
    Format: ref_segment|row_line|duncan drive|northerly
    
    Args:
        line: Reference segment LineData
        
    Returns:
        str: Reference segment fingerprint
    """
    reference = normalize_text(line.reference or '')
    monument = normalize_text(line.monument or '')
    
    # Remove pipe characters to prevent format corruption
    reference = reference.replace('|', '')
    monument = monument.replace('|', '')
    
    # Extract directional information from reference or monument
    directional = ''
    combined_text = f"{reference} {monument}".strip()
    
    directional_patterns = [
        'north', 'south', 'east', 'west',
        'northeast', 'northwest', 'southeast', 'southwest',
        'northerly', 'southerly', 'easterly', 'westerly'
    ]
    
    for direction in directional_patterns:
        if direction in combined_text:
            directional = direction
            break
    
    return f"ref_segment|{reference}|{monument}|{directional}"


def generate_ref_curve_fingerprint(line: LineData) -> str:
    """
    Generate fingerprint for reference curve
    Format: ref_curve|creek_line|along creek|easterly
    
    Args:
        line: Reference curve LineData
        
    Returns:
        str: Reference curve fingerprint
    """
    # Use same logic as ref_segment but with curve prefix
    return generate_ref_segment_fingerprint(line).replace('ref_segment|', 'ref_curve|')


def generate_schema_fingerprints(schema_output: SchemaOutput) -> List[str]:
    """
    Generate fingerprints for all lines in a schema output
    
    Args:
        schema_output: SchemaOutput to fingerprint
        
    Returns:
        List[str]: List of fingerprints, one per line
    """
    fingerprints = []
    
    for line in schema_output.lines:
        fingerprint = generate_line_fingerprint(line)
        fingerprints.append(fingerprint)
    
    return fingerprints


def compare_fingerprint_sets(actual_fingerprints: List[str], 
                           expected_fingerprints: List[str]) -> Dict[str, Any]:
    """
    Compare two sets of fingerprints and identify differences
    
    Args:
        actual_fingerprints: Fingerprints from actual extraction
        expected_fingerprints: Fingerprints from gold dataset
        
    Returns:
        Dict with comparison results
    """
    actual_set = set(actual_fingerprints)
    expected_set = set(expected_fingerprints)
    
    missing = expected_set - actual_set
    extra = actual_set - expected_set
    matching = actual_set & expected_set
    
    return {
        'total_expected': len(expected_fingerprints),
        'total_actual': len(actual_fingerprints),
        'matching_count': len(matching),
        'missing_count': len(missing),
        'extra_count': len(extra),
        'missing_fingerprints': list(missing),
        'extra_fingerprints': list(extra),
        'matching_fingerprints': list(matching),
        'accuracy': len(matching) / len(expected_set) if expected_set else 1.0,
        'precision': len(matching) / len(actual_set) if actual_set else 0.0,
        'recall': len(matching) / len(expected_set) if expected_set else 1.0
    }


def fingerprint_similarity(fp1: str, fp2: str) -> float:
    """
    Calculate similarity between two fingerprints
    
    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        
    Returns:
        float: Similarity score (0.0 to 1.0)
    """
    if fp1 == fp2:
        return 1.0
    
    # Split fingerprints into components
    parts1 = fp1.split('|')
    parts2 = fp2.split('|')
    
    # Must have same type to be similar
    if len(parts1) == 0 or len(parts2) == 0 or parts1[0] != parts2[0]:
        return 0.0
    
    # Compare components
    max_parts = max(len(parts1), len(parts2))
    matching_parts = 0
    
    for i in range(min(len(parts1), len(parts2))):
        if parts1[i] == parts2[i]:
            matching_parts += 1
    
    return matching_parts / max_parts if max_parts > 0 else 0.0


def find_similar_fingerprints(target_fingerprint: str, 
                            fingerprint_list: List[str],
                            threshold: float = 0.8) -> List[tuple]:
    """
    Find fingerprints similar to target fingerprint
    
    Args:
        target_fingerprint: Fingerprint to match against
        fingerprint_list: List of fingerprints to search
        threshold: Minimum similarity threshold
        
    Returns:
        List of (fingerprint, similarity_score) tuples
    """
    similar = []
    
    for fp in fingerprint_list:
        similarity = fingerprint_similarity(target_fingerprint, fp)
        if similarity >= threshold:
            similar.append((fp, similarity))
    
    # Sort by similarity (highest first)
    similar.sort(key=lambda x: x[1], reverse=True)
    
    return similar


def normalize_bearing_components(cardinal_ns: str, degrees: int, minutes: int, 
                               seconds: float, cardinal_ew: str) -> tuple:
    """
    Normalize bearing components for consistent fingerprinting
    
    Args:
        cardinal_ns: North/South cardinal direction
        degrees: Degrees component
        minutes: Minutes component  
        seconds: Seconds component
        cardinal_ew: East/West cardinal direction
        
    Returns:
        tuple: (normalized_ns, normalized_degrees, normalized_minutes, normalized_seconds, normalized_ew)
    """
    # Normalize cardinal directions
    ns = 'N' if (cardinal_ns or '').upper().startswith('N') else 'S'
    ew = 'E' if (cardinal_ew or '').upper().startswith('E') else 'W'
    
    # Normalize numeric components
    norm_degrees = max(0, min(359, int(degrees or 0)))
    norm_minutes = max(0, min(59, int(minutes or 0)))
    norm_seconds = max(0.0, min(59.99, round_numeric_value(seconds or 0.0, 2)))
    
    return (ns, norm_degrees, norm_minutes, norm_seconds, ew)


def extract_fingerprint_components(fingerprint: str) -> Dict[str, Any]:
    """
    Extract components from a fingerprint for analysis
    
    Args:
        fingerprint: Fingerprint string to parse
        
    Returns:
        Dict with extracted components
    """
    parts = fingerprint.split('|')
    
    if len(parts) == 0:
        return {'type': 'unknown', 'components': []}
    
    result = {
        'type': parts[0],
        'components': parts[1:] if len(parts) > 1 else [],
        'raw_fingerprint': fingerprint
    }
    
    # Parse type-specific components
    if parts[0] in ['course', 'curve'] and len(parts) >= 7:
        result.update({
            'cardinal_ns': parts[1],
            'degrees': int(parts[2]) if parts[2].isdigit() else 0,
            'minutes': int(parts[3]) if parts[3].isdigit() else 0,
            'seconds': float(parts[4]) if parts[4].replace('.', '').isdigit() else 0.0,
            'cardinal_ew': parts[5],
            'distance': float(parts[6]) if parts[6].replace('.', '').isdigit() else 0.0
        })
    elif parts[0] in ['ref_segment', 'ref_curve'] and len(parts) >= 4:
        result.update({
            'reference': parts[1],
            'monument': parts[2],
            'directional': parts[3]
        })
    
    return result


def group_fingerprints_by_type(fingerprints: List[str]) -> Dict[str, List[str]]:
    """
    Group fingerprints by their type
    
    Args:
        fingerprints: List of fingerprints to group
        
    Returns:
        Dict mapping type to list of fingerprints
    """
    groups = {}
    
    for fp in fingerprints:
        fp_type = fp.split('|')[0] if '|' in fp else 'unknown'
        
        if fp_type not in groups:
            groups[fp_type] = []
        
        groups[fp_type].append(fp)
    
    return groups


def validate_fingerprint_format(fingerprint: str) -> List[str]:
    """
    Validate fingerprint format and return any errors
    
    Args:
        fingerprint: Fingerprint to validate
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    if not fingerprint:
        errors.append("Fingerprint is empty")
        return errors
    
    if '|' not in fingerprint:
        errors.append("Fingerprint missing separator '|'")
        return errors
    
    parts = fingerprint.split('|')
    fp_type = parts[0]
    
    # Validate based on type
    if fp_type in ['course', 'curve']:
        if len(parts) != 7:
            errors.append(f"Course/curve fingerprint should have 7 parts, got {len(parts)}")
        else:
            # Validate cardinal directions
            if parts[1] not in ['N', 'S']:
                errors.append(f"Invalid cardinal NS: {parts[1]}")
            if parts[5] not in ['E', 'W']:
                errors.append(f"Invalid cardinal EW: {parts[5]}")
            
            # Validate numeric parts
            try:
                degrees = int(parts[2])
                if not (0 <= degrees <= 359):
                    errors.append(f"Degrees out of range: {degrees}")
            except ValueError:
                errors.append(f"Invalid degrees: {parts[2]}")
            
            try:
                minutes = int(parts[3])
                if not (0 <= minutes <= 59):
                    errors.append(f"Minutes out of range: {minutes}")
            except ValueError:
                errors.append(f"Invalid minutes: {parts[3]}")
            
            try:
                seconds = float(parts[4])
                if not (0 <= seconds < 60):
                    errors.append(f"Seconds out of range: {seconds}")
            except ValueError:
                errors.append(f"Invalid seconds: {parts[4]}")
            
            try:
                distance = float(parts[6])
                if distance < 0:
                    errors.append(f"Distance cannot be negative: {distance}")
            except ValueError:
                errors.append(f"Invalid distance: {parts[6]}")
    
    elif fp_type in ['ref_segment', 'ref_curve']:
        if len(parts) != 4:
            errors.append(f"Reference fingerprint should have 4 parts, got {len(parts)}")
    
    elif fp_type == 'unknown':
        # Unknown types are allowed but should be noted
        pass
    
    else:
        errors.append(f"Unknown fingerprint type: {fp_type}")
    
    return errors


class FingerprintEngine:
    """
    High-level fingerprinting engine for batch operations
    """
    
    def __init__(self):
        self.fingerprint_cache = {}
    
    def fingerprint_schema_output(self, schema_output: SchemaOutput, 
                                 use_cache: bool = True) -> List[str]:
        """
        Generate fingerprints for schema output with optional caching
        
        Args:
            schema_output: SchemaOutput to fingerprint
            use_cache: Whether to use fingerprint cache
            
        Returns:
            List of fingerprints
        """
        if use_cache:
            # Create cache key from schema content
            cache_key = hash(schema_output.to_json())
            
            if cache_key in self.fingerprint_cache:
                return self.fingerprint_cache[cache_key]
        
        fingerprints = generate_schema_fingerprints(schema_output)
        
        if use_cache:
            self.fingerprint_cache[cache_key] = fingerprints
        
        return fingerprints
    
    def compare_outputs(self, actual: SchemaOutput, expected: SchemaOutput) -> Dict[str, Any]:
        """
        Compare two schema outputs using fingerprints
        
        Args:
            actual: Actual extraction result
            expected: Expected result from gold dataset
            
        Returns:
            Dict with detailed comparison results
        """
        actual_fps = self.fingerprint_schema_output(actual)
        expected_fps = self.fingerprint_schema_output(expected)
        
        comparison = compare_fingerprint_sets(actual_fps, expected_fps)
        
        # Add bucket comparison
        comparison['bucket_match'] = actual.bucket == expected.bucket
        comparison['actual_bucket'] = actual.bucket
        comparison['expected_bucket'] = expected.bucket
        
        return comparison
    
    def clear_cache(self):
        """Clear the fingerprint cache"""
        self.fingerprint_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.fingerprint_cache),
            'total_fingerprints': sum(len(fps) for fps in self.fingerprint_cache.values())
        }