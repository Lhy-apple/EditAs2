/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:13:01 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.CheckGlobalThis;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.SyntacticScopeCreator;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CheckGlobalThis_ESTest extends CheckGlobalThis_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("dangeous use of the global 'this' object");
      Node node1 = new Node(105, 49, (-1));
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      node1.setJSDocInfo(jSDocInfo0);
      Normalize.NormalizeStatements normalize_NormalizeStatements0 = new Normalize.NormalizeStatements(compiler0, false);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, normalize_NormalizeStatements0, syntacticScopeCreator0);
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node1, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(709);
      Node node1 = new Node(105, 49, 39);
      CompilerOptions compilerOptions0 = new CompilerOptions();
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, compilerOptions0.checkGlobalNamesLevel);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, syntacticScopeCreator0);
      Node node2 = new Node(4, node0, node1, node1);
      Node node3 = compiler0.parseSyntheticCode("", "");
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node1, node3);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      Node node0 = new Node(105, 105, 105);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      node0.setJSDocInfo(jSDocInfo0);
      Node node1 = new Node(86);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null);
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      // Undeclared exception!
      try { 
        checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CheckGlobalThis", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(42, 15, 36);
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, (Node) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = new Node(86);
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, syntacticScopeCreator0);
      Node node1 = new Node(86, node0, node0, node0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node1, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = new Node(86);
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, syntacticScopeCreator0);
      Node node1 = new Node(86, node0, node0, node0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      boolean boolean1 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      assertTrue(boolean1 == boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(42, 15, 36);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      checkGlobalThis0.visit(nodeTraversal0, node0, node0);
      assertEquals(26, Node.DIRECTCALL_PROP);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(42, 15, 36);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      Node[] nodeArray0 = new Node[0];
      Node node1 = new Node(35, nodeArray0, 28, 15);
      // Undeclared exception!
      try { 
        checkGlobalThis0.visit(nodeTraversal0, node0, node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = new Node(86);
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, syntacticScopeCreator0);
      Node node1 = new Node(86, node0, node0, node0);
      checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      checkGlobalThis0.visit(nodeTraversal0, node0, node0);
      assertEquals(19, Node.LABEL_PROP);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(42, 15, 36);
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      checkGlobalThis0.visit(nodeTraversal0, node0, (Node) null);
      assertEquals(4, Node.ENUM_PROP);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(709);
      Node node1 = new Node(105, 49, 39);
      CompilerOptions compilerOptions0 = new CompilerOptions();
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, compilerOptions0.checkRequires);
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, syntacticScopeCreator0);
      Node node2 = new Node(38, node1, node0, node0);
      // Undeclared exception!
      try { 
        checkGlobalThis0.shouldTraverse(nodeTraversal0, node1, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CheckGlobalThis", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node((-995));
      Node node1 = new Node(105, 49, 42);
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, syntacticScopeCreator0);
      Node node2 = new Node(86, node1, node0, node1);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node1, node0);
      assertFalse(boolean0);
  }
}
