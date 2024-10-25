/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:38:03 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.PeepholeReplaceKnownMethods;
import com.google.javascript.jscomp.UnreachableCodeElimination;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PeepholeReplaceKnownMethods_ESTest extends PeepholeReplaceKnownMethods_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      PeepholeReplaceKnownMethods peepholeReplaceKnownMethods0 = new PeepholeReplaceKnownMethods();
      Node node0 = Node.newNumber(2812.48680608);
      Node node1 = peepholeReplaceKnownMethods0.optimizeSubtree(node0);
      assertEquals(42, Node.SIDE_EFFECT_FLAGS);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      PeepholeReplaceKnownMethods peepholeReplaceKnownMethods0 = new PeepholeReplaceKnownMethods();
      Node node0 = new Node(37, 37, 37);
      Node node1 = peepholeReplaceKnownMethods0.optimizeSubtree(node0);
      assertEquals(50, Node.FREE_CALL);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      PeepholeReplaceKnownMethods peepholeReplaceKnownMethods0 = new PeepholeReplaceKnownMethods();
      Node node0 = new Node(37);
      Node node1 = new Node(35, node0, 49, 19);
      Node node2 = new Node(37, node1);
      Node node3 = peepholeReplaceKnownMethods0.optimizeSubtree(node2);
      assertNotNull(node3);
      assertSame(node3, node2);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      PeepholeReplaceKnownMethods peepholeReplaceKnownMethods0 = new PeepholeReplaceKnownMethods();
      Node node0 = Node.newString("JSC_ILLEGAL_IMPLICIT_CAST", 37, 37);
      Node node1 = Node.newString("JSC_ILLEGAL_IMPLICIT_CAST", 37, 37);
      Node node2 = new Node(33, node0, node1, 48, 4);
      Node node3 = new Node(37, node2);
      Node node4 = peepholeReplaceKnownMethods0.optimizeSubtree(node3);
      assertNotNull(node4);
      assertEquals(Integer.MAX_VALUE, node4.getSourceOffset());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      PeepholeReplaceKnownMethods peepholeReplaceKnownMethods0 = new PeepholeReplaceKnownMethods();
      Node node0 = Node.newString("JSC_NON_GLOBAL_ERROR", 37, 37);
      Node node1 = new Node(14, 20, 39);
      Node node2 = new Node(33, node0, node1, 48, 4);
      Node node3 = new Node(37, node2);
      Node node4 = peepholeReplaceKnownMethods0.optimizeSubtree(node3);
      assertEquals(1, node4.getChildCount());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      PeepholeReplaceKnownMethods peepholeReplaceKnownMethods0 = new PeepholeReplaceKnownMethods();
      Node node0 = new Node(2247, 2247, 2247);
      Compiler compiler0 = new Compiler();
      UnreachableCodeElimination unreachableCodeElimination0 = new UnreachableCodeElimination(compiler0, false);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, unreachableCodeElimination0);
      peepholeReplaceKnownMethods0.beginTraversal(nodeTraversal0);
      Node node1 = new Node(37, node0);
      Node node2 = peepholeReplaceKnownMethods0.optimizeSubtree(node1);
      assertTrue(node2.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Node node0 = new Node(2270, 2270, 2270);
      Compiler compiler0 = new Compiler();
      AbstractCompiler.LifeCycleStage abstractCompiler_LifeCycleStage0 = AbstractCompiler.LifeCycleStage.NORMALIZED_OBFUSCATED;
      compiler0.setLifeCycleStage(abstractCompiler_LifeCycleStage0);
      Normalize.NormalizeStatements normalize_NormalizeStatements0 = new Normalize.NormalizeStatements(compiler0, true);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, normalize_NormalizeStatements0);
      PeepholeReplaceKnownMethods peepholeReplaceKnownMethods0 = new PeepholeReplaceKnownMethods();
      peepholeReplaceKnownMethods0.beginTraversal(nodeTraversal0);
      Node node1 = new Node(37, node0);
      Node node2 = peepholeReplaceKnownMethods0.optimizeSubtree(node1);
      assertEquals(Integer.MAX_VALUE, node2.getSourceOffset());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      PeepholeReplaceKnownMethods peepholeReplaceKnownMethods0 = new PeepholeReplaceKnownMethods();
      Node node0 = Node.newString("INVALID_CSS_RENAMING_MAP", 37, 37);
      node0.setType(63);
      Node node1 = new Node(33, node0, node0, 48, 4);
      Node node2 = new Node(37, node1);
      // Undeclared exception!
      try { 
        peepholeReplaceKnownMethods0.optimizeSubtree(node2);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.PeepholeReplaceKnownMethods", e);
      }
  }
}
