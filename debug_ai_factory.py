#!/usr/bin/env python3
"""Debug script to see what the AI factory is creating."""

import sys
from pathlib import Path

# Add the euchre package to the path
sys.path.insert(0, str(Path(__file__).parent))

from euchre.ai.ai_factory import AIFactory

def main():
    print("🔍 Debugging AI Factory")
    print("=" * 30)
    
    # Test creating different AI types
    ai_types = ["aggressive", "conservative", "balanced", "opportunistic"]
    
    for ai_type in ai_types:
        print(f"\n📊 Creating {ai_type} AI:")
        try:
            ai = AIFactory.create_ai_player(f"{ai_type.capitalize()}", ai_type, 0.5)
            print(f"  Type: {type(ai)}")
            print(f"  Class: {ai.__class__.__name__}")
            print(f"  Methods: {[m for m in dir(ai) if not m.startswith('_')]}")
            
            # Check if it has the new interface methods
            if hasattr(ai, 'should_order_up'):
                print(f"  ✅ Has should_order_up method")
            else:
                print(f"  ❌ Missing should_order_up method")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    main() 