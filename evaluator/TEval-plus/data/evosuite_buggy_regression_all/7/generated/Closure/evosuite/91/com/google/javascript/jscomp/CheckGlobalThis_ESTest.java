/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:11:22 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.CheckGlobalThis;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.jscomp.SyntheticAst;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CheckGlobalThis_ESTest extends CheckGlobalThis_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSType[] jSTypeArray0 = new JSType[22];
      Node node0 = jSTypeRegistry0.createParameters(jSTypeArray0);
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) checkGlobalThis0);
      assertEquals(34, Node.PARENTHESIZED_PROP);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      Node node0 = new Node(105, 39, 1);
      node0.addSuppression("+c&B{7tyi.V!\nRKMk");
      Node node1 = compiler0.parseSyntheticCode("+c&B{7tyi.V!\nRKMk", "+c&B{7tyi.V!\nRKMk");
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      Node node0 = new Node(105);
      node0.addSuppression("");
      SyntheticAst syntheticAst0 = new SyntheticAst("");
      Node node1 = syntheticAst0.getAstRoot(compiler0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      Node node0 = new Node(105, (-3520), 1);
      Node node1 = new Node(38, node0, node0, node0);
      node0.addChildrenToBack(node1);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(105);
      node0.addSuppression("");
      Node node1 = Node.newString("", 35, 29);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, (ScopeCreator) null);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      Node node0 = new Node(105, 39, 1);
      Node node1 = new Node(85, node0, 49, 18);
      Node node2 = Node.newNumber((double) 1, 41, 1000);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node2);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, (ScopeCreator) null);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSType[] jSTypeArray0 = new JSType[1];
      Node node0 = jSTypeRegistry0.createParameters(jSTypeArray0);
      Node node1 = new Node(86, node0, 26, 38);
      checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, (ScopeCreator) null);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSType[] jSTypeArray0 = new JSType[11];
      Node node0 = jSTypeRegistry0.createParameters(jSTypeArray0);
      Node node1 = new Node(86, node0, 26, 38);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node1, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = new Node(42, 42, 42);
      checkGlobalThis0.visit((NodeTraversal) null, node0, node0);
      assertEquals(25, Node.ISNUMBER_PROP);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, (ScopeCreator) null);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSType[] jSTypeArray0 = new JSType[1];
      Node node0 = jSTypeRegistry0.createParameters(jSTypeArray0);
      Node node1 = new Node(86, node0, 26, 38);
      checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      checkGlobalThis0.visit(nodeTraversal0, node0, node0);
      assertEquals(18, Node.SPECIAL_PROP_PROP);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = new Node(42, 25, 4095);
      checkGlobalThis0.visit((NodeTraversal) null, node0, (Node) null);
      assertEquals(0, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      Node node0 = new Node(105, 105, 105);
      Node node1 = new Node(86, node0, 27, 17);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      Node node0 = new Node(105, (-2384), 1);
      Node node1 = new Node(38, node0, node0, node0);
      node1.addSuppression("%wR(YFVD");
      // Undeclared exception!
      try { 
        checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, (Node) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CheckGlobalThis", e);
      }
  }
}