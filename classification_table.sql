-- Classification Data Table for Legal Description Reasoning
-- Stores AI classification reasoning data as JSONB for shared knowledge base

CREATE TABLE classification_data (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  
  -- Store the entire JSON structure from classification reasoning
  reasoning_data JSONB NOT NULL,
  
  -- Extract key fields for indexing and filtering
  classification TEXT GENERATED ALWAYS AS (reasoning_data->>'classification') STORED,
  confidence TEXT GENERATED ALWAYS AS (reasoning_data->>'confidence') STORED,
  created_at TIMESTAMPTZ GENERATED ALWAYS AS ((reasoning_data->>'timestamp')::timestamptz) STORED,
  
  -- Metadata
  inserted_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast filtering and queries
CREATE INDEX idx_classification_data_classification ON classification_data(classification);
CREATE INDEX idx_classification_data_confidence ON classification_data(confidence);  
CREATE INDEX idx_classification_data_created_at ON classification_data(created_at DESC);
CREATE INDEX idx_classification_data_inserted_at ON classification_data(inserted_at DESC);

-- GIN index for complex JSON queries
CREATE INDEX idx_classification_data_reasoning ON classification_data USING GIN (reasoning_data);

-- Optional: Add some constraints
ALTER TABLE classification_data ADD CONSTRAINT check_classification_values 
  CHECK (classification IN ('explicit_bearings', 'abstract_bearings', 'external_ref'));
  
ALTER TABLE classification_data ADD CONSTRAINT check_confidence_values 
  CHECK (confidence IN ('high', 'medium', 'low'));

-- Enable Row Level Security (optional, but good practice)
ALTER TABLE classification_data ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all authenticated users to read/write (since data is public)
CREATE POLICY "Allow all operations for authenticated users" ON classification_data
  FOR ALL USING (auth.role() = 'authenticated');

-- Allow anonymous read access (optional, for public viewing)
CREATE POLICY "Allow anonymous read access" ON classification_data
  FOR SELECT USING (true);
