#!/usr/bin/env python3
"""
Quick script to check the latest database entry for new metrics
"""
import os
import json
from supabase import create_client

def check_latest_entry():
    try:
        # Set the keys directly
        url = 
        key = 
        
        supabase = create_client(url, key)
        
        # Get the latest entry
        result = supabase.table('classification_data')\
            .select('*')\
            .order('inserted_at', desc=True)\
            .limit(1)\
            .execute()
        
        if not result.data:
            print("❌ No entries found in database")
            return
        
        entry = result.data[0]
        reasoning_data = entry.get('reasoning_data', {})
        
        print("✅ Latest database entry found!")
        print(f"📅 Inserted at: {entry.get('inserted_at')}")
        print(f"👤 User: {reasoning_data.get('user_email', 'N/A')}")
        print(f"📄 Filename: {reasoning_data.get('filename', 'N/A')}")
        
        # Check for our metrics (old + new)
        print("\n🔍 Checking metrics:")
        file_size = reasoning_data.get('file_size')
        debug_mode = reasoning_data.get('debug_mode')
        processing_time = reasoning_data.get('processing_time')
        page_count = reasoning_data.get('page_count')
        text_length = reasoning_data.get('text_length')
        bearing_count = reasoning_data.get('bearing_count')
        
        print(f"📏 file_size: {file_size} {'✅' if file_size is not None else '❌'}")
        print(f"🐛 debug_mode: {debug_mode} {'✅' if debug_mode is not None else '❌'}")
        print(f"⏱️ processing_time: {processing_time} {'✅' if processing_time is not None else '❌'}")
        print(f"📄 page_count: {page_count} {'✅' if page_count is not None else '❌'}")
        print(f"📝 text_length: {text_length} {'✅' if text_length is not None else '❌'}")
        print(f"🎯 bearing_count: {bearing_count} {'✅' if bearing_count is not None else '❌'}")
        
        parsed_bearing_count = reasoning_data.get('parsed_bearing_count')
        parsing_success_rate = reasoning_data.get('parsing_success_rate')
        print(f"📊 parsed_bearing_count: {parsed_bearing_count} {'✅' if parsed_bearing_count is not None else '❌'}")
        print(f"📈 parsing_success_rate: {parsing_success_rate}% {'✅' if parsing_success_rate is not None else '❌'}")
        
        # Show all keys in reasoning_data
        print(f"\n📋 All reasoning_data keys: {list(reasoning_data.keys())}")
        
        # Show full reasoning data if small enough
        if len(json.dumps(reasoning_data)) < 1000:
            print(f"\n📄 Full reasoning_data:\n{json.dumps(reasoning_data, indent=2)}")
        else:
            print(f"\n📄 Reasoning data too large to display ({len(json.dumps(reasoning_data))} chars)")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    check_latest_entry()
