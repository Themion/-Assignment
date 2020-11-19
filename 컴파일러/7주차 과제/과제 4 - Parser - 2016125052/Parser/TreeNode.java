package Parser;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class TreeNode implements Iterable<TreeNode>{
    private String data;
    private TreeNode parent;
    private List<TreeNode> children;
 
    public TreeNode(String data) {
        this.setData(data);
        this.children = new LinkedList<TreeNode>();
    }
 
    public TreeNode addChild(String child) {
        TreeNode childNode = new TreeNode(child);
        childNode.setParent(this);
        this.children.add(childNode);
        return childNode;
    }
 
    public TreeNode addChild(TreeNode child) {
        TreeNode childNode = new TreeNode(child.getData());
        for(TreeNode i: child.children) childNode.addChild(i);
        
        childNode.setParent(this);
        this.children.add(childNode);
        return childNode;
    }
    
    public TreeNode get(String s) {
    	for(TreeNode i : this.children)
    		if(i.getData() == s) return i;
    	
    	return null;
    }
    
    public int size() {
    	return this.children.size();
    }
    
    public TreeNode get(int i) {
    	return this.children.get(i);
    }
    
    public void remove(int i) {
    	this.children.remove(i);
    }
    
    public void popThis() {
        if (this.children.size() == 1) {
        	TreeNode c = this.children.get(0);
    		this.children.remove(0);
    		
    		this.setData(c.getData());
    		for(TreeNode i : c.children) this.addChild(i);
    	}
        else if (this.children.size() == 0) this.getParent().children.remove(this);
    }
    
    public void print(int spaces) {
    	for(int i = 0; i < spaces; i++) System.out.print("  ");
    	System.out.println(this.getData());
    	
    	for(TreeNode i : this.children) 
    		i.print(spaces + 1);
    }
    
    public void print() {
    	this.print(0);
    }
 
    @Override
    public Iterator<TreeNode> iterator() {
        // TODO Auto-generated method stub
        return null;
    }

    public static void main(String args[]) {
    	TreeNode root = new TreeNode("root"), temp = root;
    	for(int i = 0; i < 4; i++) root.addChild("" + i);
    	root.children.get(0).addChild("child");
    	root.children.get(0).children.get(0).addChild("wow");
    	root.children.get(0).children.get(0).addChild("woww");
    	root.children.get(0).children.get(0).children.get(0).addChild("amazing1");
    	root.children.get(0).children.get(0).children.get(0).addChild("amazing2");
    	
    	temp.print();
    }

	public void setParent(TreeNode parent) {
		this.parent = parent;
	}

	public TreeNode getParent() {
		return parent;
	}

	public String getData() {
		return data;
	}

	public void setData(String data) {
		this.data = data;
	}
}