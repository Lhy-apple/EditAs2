/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:09:44 GMT 2023
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
import com.google.javascript.jscomp.PeepholeRemoveDeadCode;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.jscomp.SyntacticScopeCreator;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CheckGlobalThis_ESTest extends CheckGlobalThis_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = Node.newNumber(726.4806988676);
      Node node1 = new Node(105, node0, node0, node0);
      Node node2 = new Node(47, node1, node1, 50, 31);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node1, node0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, (ScopeCreator) null);
      Node node0 = Node.newNumber(726.4806988676);
      Node node1 = new Node(105, node0, node0, node0);
      Node node2 = new Node(47, node1, node1, 50, 31);
      Node node3 = compiler0.parseTestCode("A[ least one module must be provided");
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node1, node3);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Node node0 = compiler0.parseTestCode("");
      Node node1 = new Node(105, node0, node0, node0);
      PeepholeRemoveDeadCode peepholeRemoveDeadCode0 = new PeepholeRemoveDeadCode();
      Node node2 = peepholeRemoveDeadCode0.optimizeSubtree(node1);
      Node node3 = new Node(71, node1, node2, 8, 14);
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, compilerOptions0.aggressiveVarCheck);
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, syntacticScopeCreator0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node2, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      Node node0 = Node.newNumber(0.0);
      nodeTraversal0.traverse(node0);
      assertFalse(node0.isNoSideEffectsCall());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Node node0 = Node.newNumber((-1080.871716363));
      Node node1 = new Node(86, node0, node0, 6, 6);
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, compilerOptions0.checkMissingReturn);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node1, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Node node0 = Node.newNumber((-1080.871716363));
      Node node1 = new Node(86, node0, node0, 6, 6);
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, compilerOptions0.checkGlobalThisLevel);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      checkGlobalThis0.visit(nodeTraversal0, node0, node1);
      assertEquals(18, Node.SPECIAL_PROP_PROP);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Node node0 = Node.newNumber((-1080.871716363));
      Node node1 = new Node(86, node0, node0, 6, 6);
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, compilerOptions0.checkGlobalThisLevel);
      checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node1);
      boolean boolean0 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = Node.newNumber(3962.5092911433007);
      Node node1 = new Node(42, node0, node0, 29, 50);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, (ScopeCreator) null);
      checkGlobalThis0.visit(nodeTraversal0, node1, node0);
      assertEquals(2, Node.POST_FLAG);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = Node.newNumber(3962.5092911433007);
      Node node1 = new Node(42, node0, node0, 29, 50);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, (ScopeCreator) null);
      checkGlobalThis0.visit(nodeTraversal0, node1, (Node) null);
      assertEquals(41, Node.BRACELESS_TYPE);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Node node0 = Node.newNumber(0.0);
      Node node1 = new Node(105, node0, node0, node0);
      PeepholeRemoveDeadCode peepholeRemoveDeadCode0 = new PeepholeRemoveDeadCode();
      Node node2 = peepholeRemoveDeadCode0.optimizeSubtree(node1);
      Node node3 = new Node(38, node1, node2, 8, 14);
      Node node4 = new Node(17, node3, node3, 40, (-1005));
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, compilerOptions0.reportMissingOverride);
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, syntacticScopeCreator0);
      // Undeclared exception!
      try { 
        checkGlobalThis0.shouldTraverse(nodeTraversal0, node2, (Node) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CheckGlobalThis", e);
      }
  }
}