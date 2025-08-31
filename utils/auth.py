"""
Supabase Authentication helper module using local storage
Based on working example: https://github.com/bhargavmodak/streamlit-google-oauth
"""

import os
import streamlit as st
import time
from typing import Optional, Dict, Any
from supabase import create_client, Client
from utils.st_local_storage import StLocalStorage

try:
    from streamlit_js import st_js
except ImportError:
    st.error("Please install streamlit-js: pip install streamlit-js")
    st.stop()

# Local storage instance
st_ls = StLocalStorage()

# Global Supabase client
url = os.getenv("SUPABASE_URL", "https://xvlzjyjqqgfpcxqnplds.supabase.co")
key = os.getenv("SUPABASE_ANON_KEY")

if not key:
    st.error("Missing SUPABASE_ANON_KEY environment variable. Please add it to your .env file or GitHub secrets.")
    st.stop()

try:
    supabase: Client = create_client(url, key)
except Exception as e:
    st.error(f"Failed to initialize Supabase client: {str(e)}")
    st.stop()


def nav_to(url):
    """Navigate to URL in new tab"""
    js = f'window.open("{url}", "_blank");'
    st_js(js, key="nav_to")


def get_oauth_url(provider: str = "google") -> str:
    """Get OAuth authorization URL for specified provider"""
    try:
        # Determine redirect URL based on environment
        if os.getenv("ENVIRONMENT") == "production":
            redirect_to = os.getenv("APP_URL", "https://ldr.clocknumbers.com")
        else:
            redirect_to = "http://localhost:5000"
        
        response = supabase.auth.sign_in_with_oauth({
            "provider": provider,
            "options": {"redirect_to": redirect_to}
        })
        
        return response.url
        
    except Exception as e:
        st.error(f"Failed to get OAuth URL: {str(e)}")
        return None


def show_login():
    """Show login button"""
    st.info("A new tab will open to authenticate with Google. Please close the authentication tab after logging in.")
    login_button = st.button("🟢 Login with Google", type="primary")
    if login_button:
        url = get_oauth_url("google")
        if url:
            nav_to(url)


def authenticate_user(g_session: dict):
    """Authenticate user with stored session tokens"""
    if g_session is not None:
        access_token = g_session["access_token"]
        refresh_token = g_session["refresh_token"]
        try:
            response = supabase.auth.set_session(
                access_token=access_token, refresh_token=refresh_token
            )
            return response
        except Exception as e:
            if type(e).__name__ == "AuthApiError":
                if "Invalid Refresh Token" in str(e):
                    st.error("The refresh token was already used. Please login again.")
                elif "User from sub claim in JWT does not exist" in str(e):
                    st.error("The access token was invalid. Please login again.")
                else:
                    st.error(f"Auth error: {str(e)}")
                st_ls.delete("g_session")
                st.info("Logging out...")
                if "user" in st.session_state:
                    del st.session_state["user"]
            else:
                st.error(f"Authentication error: {str(e)}")
            return None
    return None


def require_authentication():
    """Main auth function - check cookies and handle login"""
    if "user" not in st.session_state:
        with st.spinner("Checking authentication..."):
            g_session = st_ls.get("g_session")
            time.sleep(0.5)
            if g_session is None or len(g_session) == 0:
                show_login()
                st.stop()
            else:
                response = authenticate_user(g_session)
                if response is not None and response.user:
                    st.session_state.user = response.user.user_metadata
                    st.success(f"Welcome back! Signed in as {response.user.email}")
                    st.rerun()
                else:
                    show_login()
                    st.stop()
    return st.session_state.user


def show_logout():
    """Show logout button"""
    if st.button("🚪 Sign Out", use_container_width=True):
        st_ls.delete("g_session")
        st.info("Logging out...")
        st.session_state.clear()
        st.rerun()


def show_user_menu():
    """Display user info and logout option in sidebar"""
    user = st.session_state.get('user')
    
    if user:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 Account")
            
            # User avatar and info
            col1, col2 = st.columns([1, 3])
            with col1:
                if user.get('avatar_url'):
                    st.image(user['avatar_url'], width=40)
                else:
                    st.markdown("👤")
            
            with col2:
                st.write(f"**{user.get('full_name', 'User')}**")
                st.caption(user.get('email', ''))
            
            show_logout()
    
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
        
        st.info("A new tab will open to authenticate with Google. Please close the authentication tab after logging in.")
        
        # Center the login buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Sign in with:")
            
            # Google OAuth button with JavaScript popup
            google_url = self.get_oauth_url("google")
            if google_url:
                login_clicked = st.button("🟢 Continue with Google", use_container_width=True, type="primary")
                if login_clicked:
                    # Create JavaScript to open OAuth in new tab
                    oauth_js = f"""
                    <script>
                    window.open('{google_url}', 'oauth', 'width=500,height=600,scrollbars=yes,resizable=yes');
                    </script>
                    """
                    st.components.v1.html(oauth_js, height=0)
            else:
                st.error("Failed to generate Google OAuth URL")
            
            st.markdown("<br>", unsafe_allow_html=True)
        
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
