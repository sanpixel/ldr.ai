"""
Supabase Authentication helper module
Handles OAuth login, user sessions, and authentication checks
"""

import os
import streamlit as st
from typing import Optional, Dict, Any
from supabase import create_client, Client
from datetime import datetime
import json


class SupabaseAuth:
    """Handles Supabase OAuth authentication and session management"""
    
    def __init__(self):
        """Initialize Supabase client"""
        # Use environment variables with fallbacks
        self.supabase_url = os.getenv("SUPABASE_URL", "https://xvlzjyjqqgfpcxqnplds.supabase.co")
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not self.supabase_anon_key:
            st.error("Missing SUPABASE_ANON_KEY environment variable. Please add it to your .env file or GitHub secrets.")
            st.stop()
        
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_anon_key)
        except Exception as e:
            st.error(f"Failed to initialize Supabase client: {str(e)}")
            st.stop()
    
    def get_oauth_url(self, provider: str = "google") -> str:
        """Get OAuth authorization URL for specified provider"""
        try:
            # Determine redirect URL based on environment
            if os.getenv("ENVIRONMENT") == "production":
                redirect_to = os.getenv("APP_URL", "https://ldr.clocknumbers.com")
            else:
                redirect_to = "http://localhost:5000"
            
            # Generate and store code verifier for PKCE
            import secrets
            import base64
            import hashlib
            
            code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
            code_challenge = base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode('utf-8')).digest()).decode('utf-8').rstrip('=')
            
            # Store code verifier in session state for later use
            st.session_state.code_verifier = code_verifier
            
            # Also encode the code verifier in the redirect URL state parameter for persistence
            import urllib.parse
            state_data = {
                "code_verifier": code_verifier,
                "timestamp": str(int(datetime.now().timestamp()))
            }
            state_param = base64.urlsafe_b64encode(json.dumps(state_data).encode()).decode().rstrip('=')
            
            response = self.supabase.auth.sign_in_with_oauth({
                "provider": provider,
                "options": {
                    "redirect_to": redirect_to,
                    "code_challenge": code_challenge,
                    "code_challenge_method": "S256",
                    "state": state_param
                }
            })
            
            return response.url
            
        except Exception as e:
            st.error(f"Failed to get OAuth URL: {str(e)}")
            return None
    
    def handle_oauth_callback(self) -> Optional[Dict[str, Any]]:
        """Handle OAuth callback from URL parameters"""
        try:
            # Get URL parameters from Streamlit
            query_params = st.query_params
            
            # Debug: Show what parameters we received
            if query_params:
                st.info(f"🔍 DEBUG: Received OAuth callback with parameters: {list(query_params.keys())}")
            
            # Check for authorization code (standard OAuth flow)
            auth_code = query_params.get('code')
            
            if auth_code:
                st.info(f"🔄 Processing OAuth authorization code...")
                
                # Debug API key
                api_key_preview = self.supabase_anon_key[:20] + "..." if self.supabase_anon_key else "None"
                st.info(f"🔍 DEBUG: Using API key: {api_key_preview}")
                st.info(f"🔍 DEBUG: Supabase URL: {self.supabase_url}")
                
                try:
                    # Use the correct CodeExchangeParams object with code verifier
                    from supabase_auth.types import CodeExchangeParams
                    import base64
                    
                    # Get the stored code verifier from session state or URL state parameter
                    code_verifier = st.session_state.get('code_verifier')
                    
                    # If not in session state, try to get from URL state parameter
                    if not code_verifier:
                        state_param = query_params.get('state')
                        if state_param:
                            try:
                                # Decode the state parameter
                                state_data = json.loads(base64.urlsafe_b64decode(state_param + '==').decode())
                                code_verifier = state_data.get('code_verifier')
                                st.info("🔍 DEBUG: Retrieved code verifier from state parameter")
                            except Exception as state_error:
                                st.error(f"🚨 Failed to decode state parameter: {str(state_error)}")
                    
                    if not code_verifier:
                        st.error("🚨 Missing code verifier - please try signing in again")
                        st.query_params.clear()
                        return None
                    
                    code_params = CodeExchangeParams(
                        auth_code=auth_code,
                        code_verifier=code_verifier
                    )
                    
                    response = self.supabase.auth.exchange_code_for_session(code_params)
                    st.success("✅ Successfully exchanged code for session")
                    
                    # Clean up the code verifier
                    if 'code_verifier' in st.session_state:
                        del st.session_state['code_verifier']
                    
                except Exception as exchange_error:
                    st.error(f"🚨 Code exchange failed: {str(exchange_error)}")
                    st.error(f"🚨 Error type: {type(exchange_error).__name__}")
                    # Clear params and return None to stop processing
                    st.query_params.clear()
                    return None
                
                if hasattr(response, 'user') and response.user:
                    session = getattr(response, 'session', None)
                    self.store_user_session(response.user, session)
                    st.success(f"🎉 Welcome! Successfully signed in as {response.user.email}")
                    # Clear URL parameters to clean up URL
                    st.query_params.clear()
                    return response.user
                else:
                    st.warning("OAuth exchange completed but no user data received")
                    st.query_params.clear()
            
            # Fallback: Check for direct token parameters (if using implicit flow)
            access_token = query_params.get('access_token')
            refresh_token = query_params.get('refresh_token')
            
            if access_token and refresh_token:
                st.info("🔄 Processing direct tokens...")
                # Set session with tokens
                response = self.supabase.auth.set_session(access_token, refresh_token)
                
                if response.user:
                    self.store_user_session(response.user, response.session)
                    st.success(f"🎉 Welcome! Successfully signed in as {response.user.email}")
                    # Clear URL parameters to clean up URL
                    st.query_params.clear()
                    return response.user
            
            # Check for any other OAuth parameters we might have missed
            error_param = query_params.get('error')
            if error_param:
                st.error(f"OAuth error: {error_param}")
                error_description = query_params.get('error_description', 'No description provided')
                st.error(f"Error description: {error_description}")
                st.query_params.clear()
                return None
            
            # If we have unrecognized parameters, clear them
            if query_params and not auth_code and not access_token:
                st.info("🧹 Cleaning up unrecognized URL parameters")
                st.query_params.clear()
                    
            return None
            
        except Exception as e:
            st.error(f"OAuth callback error: {str(e)}")
            # Clear any problematic query params
            st.query_params.clear()
            return None
    
    def store_user_session(self, user: Dict[str, Any], session: Dict[str, Any]):
        """Store user session in Streamlit session state"""
        st.session_state.authenticated = True
        st.session_state.user = user
        st.session_state.session = session
        st.session_state.user_id = user.id
        st.session_state.user_email = user.email
        st.session_state.user_name = user.user_metadata.get('full_name') or user.user_metadata.get('name') or user.email.split('@')[0]
        st.session_state.user_avatar = user.user_metadata.get('avatar_url', '')
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user"""
        try:
            # Check session state first (fast path)
            if st.session_state.get('authenticated') and st.session_state.get('user'):
                return st.session_state.user
            
            # Try to get user from Supabase (in case of page refresh)
            response = self.supabase.auth.get_user()
            if response.user:
                session = self.supabase.auth.get_session()
                self.store_user_session(response.user, session)
                return response.user
                
            return None
            
        except Exception:
            return None
    
    def logout(self):
        """Logout user and clear session"""
        try:
            self.supabase.auth.sign_out()
        except Exception:
            pass
        
        # Clear all authentication-related session state
        auth_keys = ['authenticated', 'user', 'session', 'user_id', 
                    'user_email', 'user_name', 'user_avatar']
        for key in auth_keys:
            if key in st.session_state:
                del st.session_state[key]
    
    def require_auth(self) -> Dict[str, Any]:
        """Require authentication - show login page if not authenticated"""
        # First check for OAuth callback
        user = self.handle_oauth_callback()
        
        # If no callback, check existing session
        if not user:
            user = self.get_current_user()
        
        # If still no user, show login page
        if not user:
            self.show_login_page()
            st.stop()
            
        return user
    
    def show_login_page(self):
        """Display OAuth login page"""
        st.title("🔐 Legal Description Reader")
        st.markdown("**Welcome!** Please sign in to access your personalized legal description processing tools.")
        
        # Center the login buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Sign in with:")
            
            # Google OAuth link button
            google_url = self.get_oauth_url("google")
            if google_url:
                st.link_button("🟢 Continue with Google", google_url, use_container_width=True, type="primary")
            else:
                st.error("Failed to generate Google OAuth URL")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # GitHub OAuth link button
            github_url = self.get_oauth_url("github")
            if github_url:
                st.link_button("⚪ Continue with GitHub", github_url, use_container_width=True)
            else:
                st.error("Failed to generate GitHub OAuth URL")
        
        # App information
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✨ What You Can Do")
            st.markdown("""
            - 📄 **Upload PDF** legal descriptions
            - 🤖 **AI-powered** bearing extraction
            - 📊 **Visualize** property boundaries
            - 📁 **Export** to DXF/PDF formats
            """)
        
        with col2:
            st.markdown("### 🔒 Secure & Private")
            st.markdown("""
            - 🛡️ **Industry-standard** OAuth 2.0
            - 🔐 **Your data** stays private
            - 📈 **Track your** processing history
            - 🌐 **Access anywhere** with your account
            """)


# Convenience functions for easy use throughout the app
def get_auth_client() -> SupabaseAuth:
    """Get or create auth client singleton"""
    if 'auth_client' not in st.session_state:
        st.session_state.auth_client = SupabaseAuth()
    return st.session_state.auth_client


def require_authentication() -> Dict[str, Any]:
    """Require authentication - redirect to login if needed"""
    return get_auth_client().require_auth()


def get_current_user() -> Optional[Dict[str, Any]]:
    """Get current user without requiring authentication"""
    return get_auth_client().get_current_user()


def logout():
    """Logout current user"""
    get_auth_client().logout()


def get_supabase_client() -> Client:
    """Get the Supabase client for database operations"""
    return get_auth_client().supabase


def show_login_button():
    """Display Google sign-in button in main content area for unauthenticated users"""
    # Add custom CSS for Google-style button
    st.markdown("""
    <style>
    .google-signin-btn {
        display: inline-flex;
        align-items: center;
        background-color: #fff;
        border: 1px solid #dadce0;
        border-radius: 4px;
        color: #3c4043;
        font-family: "Google Sans", arial, sans-serif;
        font-size: 14px;
        font-weight: 500;
        padding: 12px 16px;
        text-decoration: none;
        transition: background-color 0.2s, border-color 0.2s, box-shadow 0.2s;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,.30), 0 1px 3px 1px rgba(60,64,67,.15);
        width: 100%;
        justify-content: center;
        gap: 12px;
    }
    .google-signin-btn:hover {
        background-color: #f8f9fa;
        border-color: #c1c7cd;
        box-shadow: 0 1px 3px 0 rgba(60,64,67,.30), 0 4px 8px 3px rgba(60,64,67,.15);
        text-decoration: none;
        color: #3c4043;
    }
    .google-logo {
        width: 20px;
        height: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Section header like ratio.ai
    st.markdown("<h3 style='text-align: center; margin-bottom: 24px;'>Sign in to save your work</h3>", unsafe_allow_html=True)
    
    # Create columns to center the login button
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        auth_client = get_auth_client()
        
        # Google OAuth link with custom styling
        google_url = auth_client.get_oauth_url("google")
        if google_url:
            # Custom HTML button that looks like Google's official button
            st.markdown(f"""
            <a href="{google_url}" class="google-signin-btn" target="_self">
                <svg class="google-logo" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
                Sign in with Google
            </a>
            """, unsafe_allow_html=True)
        else:
            st.error("Failed to generate Google OAuth URL")


def show_user_menu():
    """Display user info and logout option in sidebar"""
    user = get_current_user()
    
    if user:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 Account")
            
            # User avatar and info
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.session_state.get('user_avatar'):
                    st.image(st.session_state.user_avatar, width=40)
                else:
                    st.markdown("👤")
            
            with col2:
                st.write(f"**{st.session_state.get('user_name', 'User')}**")
                st.caption(st.session_state.get('user_email', ''))
            
            # Logout button
            if st.button("🚪 Sign Out", use_container_width=True):
                logout()
                st.rerun()
