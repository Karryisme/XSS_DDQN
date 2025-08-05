from html.parser import HTMLParser

class Node:
    """簡易DOM樹節點類型"""
    def __init__(self, node_type, tag=None, attrs=None, text_content=""):
        self.node_type = node_type  # 'document', 'element', or 'text'
        self.tag = tag
        self.attrs = attrs if attrs is not None else {}
        self.text_content = text_content
        self.children = []
        self.parent = None

class MyHTMLParser(HTMLParser):
    """解析HTML並構建簡單的DOM樹節點結構。"""
    def __init__(self):
        super().__init__()
        # 建立根Document節點
        self.root = Node(node_type='document', tag='#document')
        self.current_node = self.root

    def handle_starttag(self, tag, attrs):
        # 創建新元素節點並附加到當前節點
        attr_dict = {name: (value if value is not None else "") for name, value in attrs}
        new_node = Node(node_type='element', tag=tag, attrs=attr_dict)
        new_node.parent = self.current_node
        self.current_node.children.append(new_node)
        # 將當前節點下移到新節點（進入子節點）
        self.current_node = new_node

    def handle_endtag(self, tag):
        # 結束標籤，當前節點上移回父節點
        if self.current_node.parent:
            self.current_node = self.current_node.parent

    def handle_data(self, data):
        # 文本節點
        text = data.strip()
        if len(text) == 0:
            return  # 忽略純空白
        text_node = Node(node_type='text', text_content=text)
        text_node.parent = self.current_node
        self.current_node.children.append(text_node)
