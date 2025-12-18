-- Rule Version Tracking Table for Legal Description Reader
-- Tracks rule versions and their performance metrics

-- Rule version tracking
CREATE TABLE rule_versions (
  version_id TEXT PRIMARY KEY,
  gcs_path TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  performance_metrics JSONB,
  is_active BOOLEAN DEFAULT FALSE,
  
  -- Additional metadata
  created_by TEXT,
  description TEXT,
  parent_version_id TEXT,
  
  -- Performance tracking
  total_test_cases INTEGER,
  passed_test_cases INTEGER,
  failed_test_cases INTEGER,
  accuracy_percentage DECIMAL(5,2),
  
  -- Deployment tracking
  deployed_at TIMESTAMPTZ,
  rollback_count INTEGER DEFAULT 0,
  
  CONSTRAINT fk_parent_version FOREIGN KEY (parent_version_id) REFERENCES rule_versions(version_id)
);

-- Test results tracking  
CREATE TABLE harness_results (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  rule_version_id TEXT REFERENCES rule_versions(version_id),
  total_cases INTEGER NOT NULL,
  passed_cases INTEGER NOT NULL,
  failed_cases INTEGER NOT NULL,
  failure_details JSONB,
  run_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Test execution metadata
  execution_time_ms INTEGER,
  test_environment TEXT DEFAULT 'production',
  triggered_by TEXT, -- 'automatic', 'manual', 'patch_application'
  
  -- Results summary
  accuracy_percentage DECIMAL(5,2) GENERATED ALWAYS AS (
    CASE 
      WHEN total_cases > 0 THEN (passed_cases::decimal / total_cases::decimal) * 100
      ELSE 0
    END
  ) STORED
);

-- Indexes for performance
CREATE INDEX idx_rule_versions_created_at ON rule_versions(created_at DESC);
CREATE INDEX idx_rule_versions_is_active ON rule_versions(is_active);
CREATE INDEX idx_rule_versions_accuracy ON rule_versions(accuracy_percentage DESC);

CREATE INDEX idx_harness_results_version_id ON harness_results(rule_version_id);
CREATE INDEX idx_harness_results_run_at ON harness_results(run_at DESC);
CREATE INDEX idx_harness_results_accuracy ON harness_results(accuracy_percentage DESC);

-- GIN index for performance metrics JSON queries
CREATE INDEX idx_rule_versions_metrics ON rule_versions USING GIN (performance_metrics);
CREATE INDEX idx_harness_results_failure_details ON harness_results USING GIN (failure_details);

-- Constraints
ALTER TABLE rule_versions ADD CONSTRAINT check_accuracy_range 
  CHECK (accuracy_percentage >= 0 AND accuracy_percentage <= 100);

ALTER TABLE harness_results ADD CONSTRAINT check_case_counts
  CHECK (total_cases >= 0 AND passed_cases >= 0 AND failed_cases >= 0 AND 
         passed_cases + failed_cases = total_cases);

-- Enable Row Level Security
ALTER TABLE rule_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE harness_results ENABLE ROW LEVEL SECURITY;

-- Create policies for authenticated access
CREATE POLICY "Allow all operations for authenticated users" ON rule_versions
  FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON harness_results
  FOR ALL USING (auth.role() = 'authenticated');

-- Allow anonymous read access for public viewing
CREATE POLICY "Allow anonymous read access" ON rule_versions
  FOR SELECT USING (true);

CREATE POLICY "Allow anonymous read access" ON harness_results
  FOR SELECT USING (true);

-- Views for common queries
CREATE VIEW active_rule_version AS
SELECT * FROM rule_versions 
WHERE is_active = true 
ORDER BY created_at DESC 
LIMIT 1;

CREATE VIEW rule_version_summary AS
SELECT 
  rv.version_id,
  rv.created_at,
  rv.is_active,
  rv.accuracy_percentage,
  rv.total_test_cases,
  rv.rollback_count,
  COUNT(hr.id) as harness_run_count,
  MAX(hr.run_at) as last_test_run,
  AVG(hr.accuracy_percentage) as avg_test_accuracy
FROM rule_versions rv
LEFT JOIN harness_results hr ON rv.version_id = hr.rule_version_id
GROUP BY rv.version_id, rv.created_at, rv.is_active, rv.accuracy_percentage, 
         rv.total_test_cases, rv.rollback_count
ORDER BY rv.created_at DESC;