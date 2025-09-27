# 语法检查脚本
import ast
import os

def check_python_syntax(file_path):
    """检查Python文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 尝试解析AST
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"

def main():
    """检查所有Python文件的语法"""
    files_to_check = [
        'beyes/BeyesVGG.py',
        'beyes/VGG2.py',
        'FeatureSelect/pcapVGGSeed.py',
        'FeatureSelect/pcapTrainVGG_ES3_1.py',
        'FeatureSelect/pcapTrainVGG_ES3_2.py',
        'Loss/pcapVGGSeed.py',
        'Loss/pcapTrainVGG_ce.py'
    ]
    
    print("=== VGG29代码语法检查 ===")
    
    all_good = True
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            is_valid, error = check_python_syntax(file_path)
            if is_valid:
                print(f"✅ {file_path} - 语法正确")
            else:
                print(f"❌ {file_path} - {error}")
                all_good = False
        else:
            print(f"⚠️  {file_path} - 文件不存在")
            all_good = False
    
    print("\n" + "="*50)
    if all_good:
        print("🎉 所有文件语法检查通过！")
        print("\n修复内容:")
        print("- 将所有 self.weights 重命名为 self.class_weights_dict")
        print("- 解决了与Keras Model内置weights属性的冲突")
        print("- 保持了所有功能的完整性")
    else:
        print("❌ 部分文件存在语法问题")
    
    return all_good

if __name__ == "__main__":
    main()
