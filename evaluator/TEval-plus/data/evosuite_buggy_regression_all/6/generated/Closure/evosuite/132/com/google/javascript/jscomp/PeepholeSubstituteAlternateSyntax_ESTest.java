/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:04:28 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.PeepholeRemoveDeadCode;
import com.google.javascript.jscomp.PeepholeSubstituteAlternateSyntax;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PeepholeSubstituteAlternateSyntax_ESTest extends PeepholeSubstituteAlternateSyntax_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Node node0 = new Node(3346);
      Node node1 = new Node(4, node0, node0, 40, 4095);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Node node0 = new Node(1077, 1077, 1077);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertEquals(49, Node.DIRECT_EVAL);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Node node0 = new Node(26);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.PeepholeSubstituteAlternateSyntax", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Node node0 = new Node(30, 30, 30);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Node node0 = new Node(37);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Node node0 = new Node(38, 38, 38);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Node node0 = Node.newString(43, "com.google.javascript.jscomp.parsing.ParserRunner$ParseResult");
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.PeepholeSubstituteAlternateSyntax", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Node node0 = new Node(44);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.PeepholeSubstituteAlternateSyntax", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Node node0 = new Node(49, 49, 49);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Node node0 = new Node(63);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertFalse(node1.isCatch());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Node node0 = new Node(114, 114, 114);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.PeepholeSubstituteAlternateSyntax", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Node node0 = new Node(115, 115, 115);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // malformed 'for' statement FOR 115
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Node node0 = new Node(130);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.PeepholeSubstituteAlternateSyntax", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Node node0 = new Node(115);
      Node node1 = Node.newNumber(0.0);
      node0.addChildToFront(node1);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // malformed 'for' statement FOR
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Node node0 = new Node(115);
      node0.addChildToFront(node0);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // malformed 'for' statement FOR
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Node node0 = new Node(85);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertFalse(node1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Node node0 = new Node(85);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Node node1 = new Node(31, node0, node0, node0, node0);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertNotNull(node2);
      assertFalse(node2.hasChildren());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Node node0 = new Node(125);
      Node node1 = new Node(125, node0, node0);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
      assertFalse(node2.hasMoreThanOneChild());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Node node0 = new Node(3346);
      Node node1 = new Node(4, node0, node0, 40, 4095);
      node1.addChildToFront(node1);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      // Undeclared exception!
      peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Node node0 = new Node(4);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      boolean boolean0 = peepholeSubstituteAlternateSyntax0.isPure((Node) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Node node0 = new Node(37);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      boolean boolean0 = peepholeSubstituteAlternateSyntax0.isPure(node0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Node node0 = new Node(21, 21, 21);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      boolean boolean0 = peepholeSubstituteAlternateSyntax0.isPure(node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Node node0 = Node.newString(0, "");
      boolean boolean0 = peepholeSubstituteAlternateSyntax0.isPure(node0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Node node0 = new Node(4, 4, 4);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      boolean boolean0 = peepholeSubstituteAlternateSyntax0.areMatchingExits(node0, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Node node0 = new Node(3360);
      Node node1 = new Node(4, node0, node0, 32, 4095);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.areMatchingExits(node1, node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ControlFlowAnalysis", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Node node0 = new Node(30);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.isExceptionPossible(node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Node node0 = Node.newString(49, "");
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      boolean boolean0 = peepholeSubstituteAlternateSyntax0.isExceptionPossible(node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Node node0 = new Node(1077, 1077, 1077);
      Node node1 = new Node(85);
      PeepholeRemoveDeadCode peepholeRemoveDeadCode0 = new PeepholeRemoveDeadCode();
      Node node2 = peepholeRemoveDeadCode0.tryOptimizeBlock(node1);
      Node node3 = new Node(108, node2, node0);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node4 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node3);
      assertEquals(12, Node.COLUMN_BITS);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Node node0 = new Node(98);
      Node node1 = new Node(98, node0, node0);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.PeepholeSubstituteAlternateSyntax", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node0 = Node.newNumber((double) 40);
      Node node1 = new Node(113, node0, node0);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Node node0 = new Node(44, 44, 44);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertEquals(55, Node.LAST_PROP);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Node node0 = new Node(1077);
      Node node1 = new Node(63, node0, node0);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
      assertEquals(54, Node.SLASH_V);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Node node0 = new Node(63, 63, 63);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertFalse(node1.isFalse());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      boolean boolean0 = PeepholeSubstituteAlternateSyntax.containsUnicodeEscape("v&z4/(Fc8)6,O{");
      assertTrue(boolean0);
  }
}