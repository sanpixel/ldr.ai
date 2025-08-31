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
    """Navigate to URL in popup window"""
    js = f'window.open("{url}", "auth_popup", "width=500,height=600,scrollbars=yes,resizable=yes");'
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
    """Display clean Google-style user button in sidebar"""
    user = st.session_state.get('user')
    
    if user:
        with st.sidebar:
            st.markdown("---")
            
            # Custom CSS for Google-style user button
            st.markdown("""
            <style>
            .user-button {
                display: flex;
                align-items: center;
                background-color: #fff;
                border: 1px solid #dadce0;
                border-radius: 20px;
                color: #3c4043;
                font-family: "Google Sans", arial, sans-serif;
                font-size: 14px;
                font-weight: 500;
                padding: 8px 12px;
                margin: 8px 0;
                box-shadow: 0 1px 2px 0 rgba(60,64,67,.30), 0 1px 3px 1px rgba(60,64,67,.15);
                width: 100%;
                gap: 8px;
                cursor: pointer;
                transition: background-color 0.2s;
            }
            .user-button:hover {
                background-color: #f8f9fa;
            }
            .user-avatar {
                width: 28px;
                height: 28px;
                border-radius: 50%;
                background-color: #1a73e8;
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                font-weight: 500;
            }
            .user-info {
                flex: 1;
                overflow: hidden;
            }
            .user-name {
                font-weight: 500;
                color: #3c4043;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .user-email {
                font-size: 12px;
                color: #5f6368;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # User button display
            user_name = user.get('full_name', 'User')
            user_email = user.get('email', '')
            avatar_url = user.get('avatar_url')
            
            if avatar_url:
                avatar_html = f'<img src="{avatar_url}" class="user-avatar" alt="Avatar">'
            else:
                # Use initials as fallback
                initials = ''.join([name[0].upper() for name in user_name.split()[:2]])
                avatar_html = f'<div class="user-avatar">{initials}</div>'
            
            st.markdown(f"""
            <div class="user-button">
                {avatar_html}
                <div class="user-info">
                    <div class="user-name">{user_name}</div>
                    <div class="user-email">{user_email}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            show_logout()


# Simplified convenience functions
def get_current_user() -> Optional[Dict[str, Any]]:
    """Get current user from session state"""
    return st.session_state.get('user')


def get_supabase_client() -> Client:
    """Get the Supabase client for database operations"""
    return supabase


def show_login_button():
    """Display Google sign-in button for unauthenticated users"""
    # Add custom CSS for compact Google-style button
    st.markdown("""
    <style>
    .google-signin-btn-compact {
        display: inline-flex;
        align-items: center;
        background-color: #fff;
        border: 1px solid #dadce0;
        border-radius: 4px;
        color: #3c4043;
        font-family: "Google Sans", arial, sans-serif;
        font-size: 12px;
        font-weight: 500;
        padding: 8px 12px;
        text-decoration: none;
        transition: background-color 0.2s, border-color 0.2s, box-shadow 0.2s;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,.30), 0 1px 3px 1px rgba(60,64,67,.15);
        width: 100%;
        justify-content: center;
        gap: 8px;
        margin: 8px 0;
    }
    .google-signin-btn-compact:hover {
        background-color: #f8f9fa;
        border-color: #c1c7cd;
        box-shadow: 0 1px 3px 0 rgba(60,64,67,.30), 0 4px 8px 3px rgba(60,64,67,.15);
        text-decoration: none;
        color: #3c4043;
    }
    .google-logo-compact {
        width: 16px;
        height: 16px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Google OAuth link with compact styling
    google_url = get_oauth_url("google")
    if google_url:
        # Custom HTML button that looks like Google's official button but compact
        st.markdown(f"""
        <a href="{google_url}" class="google-signin-btn-compact" target="_self">
            <svg class="google-logo-compact" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
            </svg>
            Sign in
        </a>
        """, unsafe_allow_html=True)
    else:
        st.error("Failed to generate Google OAuth URL")


