/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:13:06 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.jscomp.CheckGlobalThis;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CheckGlobalThis_ESTest extends CheckGlobalThis_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      Node node0 = new Node(105);
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node1 = new Node(28, node0, node0, 20, 37);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      Node node0 = new Node(105);
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      node0.setJSDocInfo(jSDocInfo0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      Node node0 = Node.newString("");
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      nodeTraversal0.traverse(node0);
      assertFalse(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      Node node0 = new Node(32, 32, 32);
      Node node1 = new Node(86, node0, node0, node0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node1, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      Node node0 = new Node(32, 32, 32);
      Node node1 = new Node(86, node0, node0, node0);
      checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      checkGlobalThis0.visit(nodeTraversal0, node0, node0);
      assertEquals(41, Node.BRACELESS_TYPE);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      Node node0 = Node.newString(")GI`_V~?[~T)", 294, 294);
      Node node1 = new Node(86, node0, node0, node0);
      checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      Node[] nodeArray0 = new Node[0];
      Node node0 = new Node(42, nodeArray0, 42, 42);
      checkGlobalThis0.visit(nodeTraversal0, node0, node0);
      assertEquals(11, Node.USES_PROP);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      Node node0 = new Node(105);
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node1 = new Node(86, node0, node0, 36, 24);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      Node node0 = new Node(105);
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node1 = new Node(38, node0, node0, 21, 1);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      node1.setJSDocInfo(jSDocInfo0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      Node node0 = new Node(105);
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node1 = new Node(38, node0, node0, 21, 1);
      Node node2 = new Node(28, node1, node1, 20, 37);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node0);
      assertTrue(boolean0);
  }
}