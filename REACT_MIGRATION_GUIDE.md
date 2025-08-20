# React Frontend Migration Guide for LDR.ai

## Executive Summary

This document outlines the complete migration strategy from the current Streamlit frontend to a React-based frontend while maintaining the Python backend as an API service on Google Cloud Run.

## Migration Benefits

### User Experience Improvements
- **Faster Loading**: React apps load once, subsequent interactions are instantaneous
- **Better Mobile Support**: Native responsive design capabilities
- **Improved UI/UX**: More control over styling, animations, and interactions  
- **Real-time Feedback**: Better progress indicators and loading states
- **Offline Capabilities**: Potential for service worker implementation

### Developer Experience
- **Modern Development**: Industry-standard React development patterns
- **Better Testing**: Jest, React Testing Library ecosystem
- **Component Reusability**: Modular, reusable UI components
- **State Management**: Redux/Context API for complex state
- **Hot Reloading**: Faster development iteration

### Performance & Scalability
- **Client-side Rendering**: Reduces server load
- **Caching**: Better browser caching strategies
- **CDN Distribution**: Static assets can be served from CDN
- **Progressive Enhancement**: Core functionality works without JavaScript

## Architecture Overview

### Current Architecture (Streamlit)
```
User Browser ←→ Streamlit App (Frontend + Backend) ←→ OpenAI API
                     ↓
               Google Cloud Run
```

### Proposed Architecture (React + API)
```
User Browser ←→ React App (Static/CDN) ←→ Python FastAPI ←→ OpenAI API
                                              ↓
                                        Google Cloud Run
                                              ↓
                                         Supabase DB
```

## Migration Strategy

### Phase 1: Backend API Development (Week 1-2)

#### 1.1 Create FastAPI Backend
- **Framework**: FastAPI (Python) for API performance and automatic documentation
- **File Structure**:
  ```
  backend/
  ├── app/
  │   ├── main.py           # FastAPI app entry point
  │   ├── api/
  │   │   ├── endpoints/
  │   │   │   ├── pdf.py    # PDF processing endpoints
  │   │   │   ├── bearings.py # Bearing extraction
  │   │   │   └── export.py # DXF/PDF export
  │   │   └── dependencies.py
  │   ├── core/
  │   │   ├── config.py     # Environment configuration
  │   │   └── security.py   # Authentication
  │   ├── models/
  │   │   ├── bearing.py    # Pydantic models
  │   │   └── pdf.py
  │   ├── services/
  │   │   ├── gpt_service.py
  │   │   ├── pdf_service.py
  │   │   └── export_service.py
  │   └── utils/
  │       ├── calculations.py
  │       └── file_helpers.py
  ├── requirements.txt
  ├── Dockerfile
  └── cloudbuild.yaml
  ```

#### 1.2 API Endpoints Design
```python
# Core Endpoints
POST /api/v1/pdf/upload          # Upload and process PDF
POST /api/v1/bearings/extract    # Extract bearings from text
GET  /api/v1/bearings/{id}       # Get extracted bearings
PUT  /api/v1/bearings/{id}       # Update bearings manually
POST /api/v1/export/dxf          # Generate DXF file
POST /api/v1/export/pdf          # Generate PDF report

# Authentication (Future)
POST /api/v1/auth/google         # Google OAuth
POST /api/v1/auth/refresh        # Token refresh
GET  /api/v1/auth/me             # Current user

# Health & Debug
GET  /api/health                 # Health check
GET  /api/debug/info             # Debug information (if DEBUG=true)
```

#### 1.3 Data Models (Pydantic)
```python
class BearingData(BaseModel):
    cardinal_ns: Literal["North", "South"]
    degrees: int = Field(ge=0, le=90)
    minutes: int = Field(ge=0, le=59)
    seconds: int = Field(ge=0, le=59)
    cardinal_ew: Literal["East", "West"]
    distance: float = Field(gt=0)
    monument: Optional[str] = ""
    original_text: str

class ProcessedPDF(BaseModel):
    id: str
    extracted_text: str
    bearings: List[BearingData]
    property_info: Optional[PropertyInfo]
    created_at: datetime
    
class ExportRequest(BaseModel):
    bearings: List[BearingData]
    format: Literal["dxf", "pdf"]
    property_info: Optional[PropertyInfo]
```

### Phase 2: React Frontend Development (Week 2-4)

#### 2.1 Technology Stack
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite (faster than Create React App)
- **Styling**: Tailwind CSS + Headless UI
- **State Management**: Zustand (lighter than Redux)
- **HTTP Client**: Axios with interceptors
- **File Upload**: React Dropzone
- **Forms**: React Hook Form + Zod validation
- **Charts**: Plotly.js React wrapper
- **Testing**: Jest + React Testing Library

#### 2.2 Component Structure
```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/                 # Reusable UI components
│   │   │   ├── Button.tsx
│   │   │   ├── Input.tsx
│   │   │   ├── Modal.tsx
│   │   │   └── Table.tsx
│   │   ├── pdf/
│   │   │   ├── PDFUploader.tsx
│   │   │   ├── PDFPreview.tsx
│   │   │   └── TextExtractor.tsx
│   │   ├── bearings/
│   │   │   ├── BearingsTable.tsx
│   │   │   ├── BearingInput.tsx
│   │   │   └── BearingsList.tsx
│   │   ├── plot/
│   │   │   ├── PlotViewer.tsx
│   │   │   └── PlotControls.tsx
│   │   └── debug/
│   │       ├── DebugPanel.tsx
│   │       └── LogViewer.tsx
│   ├── pages/
│   │   ├── HomePage.tsx
│   │   ├── ProcessPage.tsx
│   │   └── ResultsPage.tsx
│   ├── hooks/
│   │   ├── useAPI.ts
│   │   ├── useBearings.ts
│   │   └── useDebounce.ts
│   ├── stores/
│   │   ├── bearingsStore.ts
│   │   ├── uiStore.ts
│   │   └── debugStore.ts
│   ├── types/
│   │   ├── bearing.ts
│   │   ├── api.ts
│   │   └── pdf.ts
│   ├── utils/
│   │   ├── calculations.ts
│   │   ├── formatters.ts
│   │   └── validators.ts
│   └── api/
│       ├── client.ts
│       ├── endpoints/
│       │   ├── pdf.ts
│       │   ├── bearings.ts
│       │   └── export.ts
│       └── types.ts
├── public/
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```

#### 2.3 Key Features Implementation

##### PDF Processing Flow
```typescript
// Simplified workflow
const processPDF = async (file: File) => {
  // 1. Upload PDF
  const uploadResponse = await api.pdf.upload(file);
  
  // 2. Show preview immediately
  setPdfPreview(uploadResponse.preview_url);
  
  // 3. Extract text (with progress)
  const extractResponse = await api.pdf.extractText(uploadResponse.id);
  setExtractedText(extractResponse.text);
  
  // 4. Process bearings (with loading state)
  const bearingsResponse = await api.bearings.extract({
    text: extractResponse.text,
    classification: true
  });
  
  setBearings(bearingsResponse.bearings);
  setClassification(bearingsResponse.classification);
};
```

##### State Management (Zustand)
```typescript
interface BearingsStore {
  bearings: BearingData[];
  isLoading: boolean;
  error: string | null;
  
  // Actions
  setBearings: (bearings: BearingData[]) => void;
  updateBearing: (id: string, updates: Partial<BearingData>) => void;
  addBearing: () => void;
  removeBearing: (id: string) => void;
  
  // Async actions
  extractFromPDF: (file: File) => Promise<void>;
  saveBearings: () => Promise<void>;
}
```

##### Debug Panel Component
```typescript
const DebugPanel = () => {
  const { debugEnabled } = useDebugStore();
  const [logs, setLogs] = useState<LogEntry[]>([]);
  
  if (!debugEnabled) return null;
  
  return (
    <Collapsible>
      <CollapsibleTrigger>🔍 Debug Information</CollapsibleTrigger>
      <CollapsibleContent>
        <Tabs>
          <TabsList>
            <TabsTrigger value="classification">AI Classification</TabsTrigger>
            <TabsTrigger value="ocr">OCR Results</TabsTrigger>
            <TabsTrigger value="logs">System Logs</TabsTrigger>
          </TabsList>
          
          <TabsContent value="classification">
            <ClassificationDebug />
          </TabsContent>
          
          <TabsContent value="ocr">
            <OCRDebug />
          </TabsContent>
          
          <TabsContent value="logs">
            <LogViewer logs={logs} />
          </TabsContent>
        </Tabs>
      </CollapsibleContent>
    </Collapsible>
  );
};
```

### Phase 3: Data Persistence with Supabase (Week 3)

#### 3.1 Database Schema
```sql
-- Users table (for future OAuth)
CREATE TABLE users (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- PDF Processing Sessions
CREATE TABLE pdf_sessions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  filename TEXT NOT NULL,
  extracted_text TEXT,
  pdf_image_url TEXT,
  property_info JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Bearings Data
CREATE TABLE bearings (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id UUID REFERENCES pdf_sessions(id),
  cardinal_ns TEXT NOT NULL CHECK (cardinal_ns IN ('North', 'South')),
  degrees INTEGER NOT NULL CHECK (degrees >= 0 AND degrees <= 90),
  minutes INTEGER NOT NULL CHECK (minutes >= 0 AND minutes <= 59),
  seconds INTEGER NOT NULL CHECK (seconds >= 0 AND seconds <= 59),
  cardinal_ew TEXT NOT NULL CHECK (cardinal_ew IN ('East', 'West')),
  distance FLOAT NOT NULL CHECK (distance > 0),
  monument TEXT DEFAULT '',
  original_text TEXT,
  sequence_order INTEGER NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Classification Reasoning (for debugging)
CREATE TABLE classification_logs (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id UUID REFERENCES pdf_sessions(id),
  classification TEXT NOT NULL,
  confidence TEXT,
  reasoning TEXT,
  evidence TEXT,
  alternatives JSONB,
  full_response TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_bearings_session_id ON bearings(session_id);
CREATE INDEX idx_bearings_sequence ON bearings(session_id, sequence_order);
CREATE INDEX idx_classification_logs_session_id ON classification_logs(session_id);
```

#### 3.2 Supabase Integration
```typescript
// Backend service
class SupabaseService {
  private supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY);
  
  async savePDFSession(sessionData: PDFSessionData): Promise<string> {
    const { data, error } = await this.supabase
      .from('pdf_sessions')
      .insert(sessionData)
      .select()
      .single();
    
    if (error) throw error;
    return data.id;
  }
  
  async saveBearings(sessionId: string, bearings: BearingData[]): Promise<void> {
    const bearingsWithSession = bearings.map((bearing, index) => ({
      ...bearing,
      session_id: sessionId,
      sequence_order: index
    }));
    
    const { error } = await this.supabase
      .from('bearings')
      .upsert(bearingsWithSession);
    
    if (error) throw error;
  }
  
  async saveClassificationLog(sessionId: string, logData: ClassificationLog): Promise<void> {
    const { error } = await this.supabase
      .from('classification_logs')
      .insert({ ...logData, session_id: sessionId });
    
    if (error) throw error;
  }
}
```

### Phase 4: Deployment & Infrastructure (Week 4)

#### 4.1 Backend Deployment (Google Cloud Run)
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/ldr-api', './backend']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/ldr-api']
  
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'ldr-api'
      - '--image=gcr.io/$PROJECT_ID/ldr-api'
      - '--region=us-central1'
      - '--platform=managed'
      - '--allow-unauthenticated'
      - '--memory=2Gi'
      - '--cpu=2'
      - '--max-instances=10'
      - '--set-env-vars=DEBUG_MODE=false,OPENAI_API_KEY=${_OPENAI_API_KEY}'
```

#### 4.2 Frontend Deployment (Vercel/Netlify)
```json
// vercel.json
{
  "version": 2,
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build"
    }
  ],
  "routes": [
    { "src": "/api/(.*)", "dest": "https://ldr-api-xxx.run.app/api/$1" },
    { "src": "/(.*)", "dest": "/index.html" }
  ],
  "env": {
    "VITE_API_URL": "https://ldr-api-xxx.run.app",
    "VITE_DEBUG_MODE": "false"
  }
}
```

#### 4.3 Environment Configuration
```typescript
// Frontend environment config
export const config = {
  apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  debugMode: import.meta.env.VITE_DEBUG_MODE === 'true',
  maxFileSize: 10 * 1024 * 1024, // 10MB
  supportedFileTypes: ['application/pdf'],
  version: import.meta.env.VITE_APP_VERSION || '2.0.0'
};

// Backend environment config  
class Settings:
    debug_mode: bool = Field(default=False)
    openai_api_key: str = Field(...)
    supabase_url: str = Field(...)
    supabase_service_key: str = Field(...)
    cors_origins: List[str] = Field(default=["http://localhost:5173"])
    max_file_size: int = Field(default=10_000_000)  # 10MB
```

## Migration Timeline

### Week 1: Backend API Foundation
- [ ] Set up FastAPI project structure
- [ ] Implement core PDF processing endpoints
- [ ] Migrate GPT integration to API service
- [ ] Set up basic error handling and logging
- [ ] Create Docker configuration for Cloud Run
- [ ] Test API endpoints with Postman/curl

### Week 2: React Frontend Setup  
- [ ] Initialize React + TypeScript + Vite project
- [ ] Set up Tailwind CSS and component library
- [ ] Create basic routing and page structure
- [ ] Implement PDF upload component
- [ ] Build bearings table with editing capabilities
- [ ] Integrate Plotly for line drawing visualization

### Week 3: Integration & Data Persistence
- [ ] Connect React frontend to FastAPI backend
- [ ] Set up Supabase database and tables
- [ ] Implement data persistence in backend
- [ ] Add debug panel with expandable sections
- [ ] Create export functionality (DXF/PDF)
- [ ] Add loading states and error handling

### Week 4: Testing & Deployment
- [ ] Write unit tests for critical functions
- [ ] Set up integration tests for API endpoints
- [ ] Deploy backend to Google Cloud Run
- [ ] Deploy frontend to Vercel/Netlify
- [ ] Configure CORS and security headers
- [ ] Set up monitoring and logging
- [ ] Performance testing and optimization

## Risk Assessment & Mitigation

### High Risk Items
1. **GPT Integration Complexity**
   - Risk: Complex prompt handling and response parsing
   - Mitigation: Thoroughly test with existing prompts, maintain backward compatibility

2. **File Upload Size Limits**
   - Risk: Large PDFs may fail to process
   - Mitigation: Implement chunked uploads, compress images before storage

3. **CORS Issues**
   - Risk: Cross-origin requests may fail
   - Mitigation: Proper CORS configuration, test in production environment

### Medium Risk Items
1. **State Management Complexity**
   - Risk: React state management becomes unwieldy
   - Mitigation: Use Zustand for simple, predictable state updates

2. **Mobile Responsiveness**
   - Risk: Complex table editing on mobile devices
   - Mitigation: Design mobile-first, test on various screen sizes

### Low Risk Items
1. **TypeScript Learning Curve**
   - Risk: Development slowdown due to type definitions
   - Mitigation: Start with basic types, add complexity gradually

## Testing Strategy

### Backend Testing
```python
# pytest configuration
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_pdf_upload():
    with open("test_files/sample.pdf", "rb") as f:
        response = client.post("/api/v1/pdf/upload", files={"file": f})
    assert response.status_code == 200
    assert "id" in response.json()

def test_bearings_extraction():
    test_text = "North 45 degrees 30 minutes East 150.0 feet to an iron pin"
    response = client.post("/api/v1/bearings/extract", json={"text": test_text})
    assert response.status_code == 200
    bearings = response.json()["bearings"]
    assert len(bearings) == 1
    assert bearings[0]["degrees"] == 45
```

### Frontend Testing
```typescript
// React Testing Library
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BearingsTable } from '../components/bearings/BearingsTable';

test('allows editing bearing values', async () => {
  const mockBearings = [
    { id: '1', cardinal_ns: 'North', degrees: 45, minutes: 30, seconds: 0, cardinal_ew: 'East', distance: 150, monument: '' }
  ];
  
  render(<BearingsTable bearings={mockBearings} onUpdate={jest.fn()} />);
  
  const degreesInput = screen.getByDisplayValue('45');
  fireEvent.change(degreesInput, { target: { value: '50' } });
  
  await waitFor(() => {
    expect(screen.getByDisplayValue('50')).toBeInTheDocument();
  });
});
```

## Success Metrics

### Performance Metrics
- [ ] Initial page load: < 2 seconds
- [ ] PDF processing: < 10 seconds for typical documents
- [ ] API response time: < 500ms for non-processing endpoints
- [ ] Plot rendering: < 1 second for typical property sizes

### User Experience Metrics
- [ ] Mobile usability score: > 90 (Google PageSpeed)
- [ ] Accessibility score: > 95 (WAVE/axe testing)
- [ ] Error rate: < 1% for successful uploads
- [ ] User task completion: > 95% for core workflows

### Technical Metrics
- [ ] Test coverage: > 80% for critical paths
- [ ] Build time: < 2 minutes for full deployment
- [ ] Zero-downtime deployments
- [ ] API uptime: > 99.5%

## Questions for Simon

1. **Timeline Preferences**: Is the 4-week timeline realistic for your needs, or would you prefer a faster/phased approach?

2. **Deployment Preferences**: Would you prefer Vercel, Netlify, or Google Cloud Storage for the React frontend?

3. **Authentication Priority**: Should we implement Google OAuth during migration or defer it to a later release?

4. **Debug Features**: What specific debug information do you find most valuable in the current Streamlit version?

5. **Mobile Requirements**: How important is mobile functionality? Should we optimize for phone/tablet use?

6. **Legacy Support**: Do you need to maintain the Streamlit version during migration as a fallback?

7. **Data Migration**: Any existing user data or sessions that need to be preserved?

8. **Third-party Integrations**: Any specific requirements for analytics, monitoring, or other services?

## Next Steps

1. **Review this document** and provide feedback on approach and timeline
2. **Prioritize features** - which components are most critical for launch?
3. **Set up development environment** - confirm access to all required services
4. **Create project repositories** - separate repos for frontend/backend or monorepo?
5. **Define API contracts** - detailed specification for all endpoints
6. **Begin Phase 1** - FastAPI backend development

This migration will significantly improve the user experience while maintaining all current functionality and adding better debugging capabilities, data persistence, and mobile support.
