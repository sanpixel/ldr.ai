# Local Development Setup

## Quick Start

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Fill in your API keys in `.env`:**
   - Get OpenAI API key from https://platform.openai.com/api-keys
   - Get Supabase keys from your Supabase project dashboard
   - Get Google OAuth credentials from Google Cloud Console
   - (Optional) Get Google Drive API key for Drive integration

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run locally:**
   ```bash
   streamlit run main.py
   ```

## Environment Variables

The app checks for environment variables in this order:

1. **Local `.env` file** (for development)
2. **System environment variables** (for production)
3. **Legacy JSON file fallback** (existing `C:\dev\openai-key.json`)

## Production vs Development

- **Production**: Uses GitHub Secrets → Cloud Run environment variables
- **Development**: Uses local `.env` file or legacy JSON file
- **Fallback**: Always maintains backward compatibility

## Important Notes

- ✅ **Production deployment stays working** - no changes to existing Cloud Run setup
- ✅ **Legacy local setup stays working** - existing JSON file method still works
- ✅ **New local setup available** - cleaner `.env` file approach for new developers
- ✅ **Cross-platform** - `.env` works on Windows, Mac, Linux
- ✅ **Secure** - `.env` file is in `.gitignore` so secrets don't get committed

## Troubleshooting

If you get API key errors:
1. Check your `.env` file has the correct keys
2. Verify the `.env` file is in the project root
3. Restart Streamlit after changing `.env`
4. Fallback: Use the existing `C:\dev\openai-key.json` method