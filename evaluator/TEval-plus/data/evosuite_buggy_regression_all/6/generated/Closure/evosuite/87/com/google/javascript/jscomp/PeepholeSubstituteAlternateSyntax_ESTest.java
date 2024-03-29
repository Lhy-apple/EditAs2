/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:59:39 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ConstCheck;
import com.google.javascript.jscomp.ControlFlowAnalysis;
import com.google.javascript.jscomp.FlowSensitiveInlineVariables;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.PeepholeSubstituteAlternateSyntax;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.jscomp.SyntacticScopeCreator;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PeepholeSubstituteAlternateSyntax_ESTest extends PeepholeSubstituteAlternateSyntax_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.SourceExcerptProvider$1");
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertEquals(27, Node.SPECIALCALL_PROP);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
      Node node0 = Node.newNumber((double) 113, 113, 113);
      Node node1 = new Node(113, node0, node0, node0, node0);
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
  public void test02()  throws Throwable  {
      Node node0 = new Node(114);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
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
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
      Node node0 = new Node(115);
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
  public void test04()  throws Throwable  {
      Node node0 = new Node(130);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
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
  public void test05()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
      Compiler compiler0 = new Compiler();
      ConstCheck constCheck0 = new ConstCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, constCheck0);
      peepholeSubstituteAlternateSyntax0.beginTraversal(nodeTraversal0);
      Node node0 = new Node(30, 30, 30);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertEquals(42, Node.IS_CONSTANT_NAME);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
      Node node0 = Node.newString("[");
      Node node1 = new Node(4, node0, node0, node0, node0);
      Node node2 = Node.newString("[");
      Node node3 = new Node(115, node1, node2, node2, (-2886), 26);
      Node node4 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
      assertEquals(15, Node.CASEARRAY_PROP);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
      Node node0 = Node.newString("[");
      Node node1 = new Node(4, node0, node0, node0, node0);
      node1.removeChildren();
      Node node2 = new Node(115, node1, node0, node0, (-2886), 26);
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
  public void test08()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.SourceEcerptProvider$1");
      Node node1 = new Node(26, node0, node0, node0, node0, 33, 49);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
      assertNotNull(node2);
      assertEquals(33, node2.getLineno());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("5ZsHgkDg[q", "5ZsHgkDg[q");
      Node node1 = new Node(4095, 16, 4095);
      Node node2 = new Node(108, node0, node0, node0, node1, 31, (-1));
      Node node3 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node2);
      assertEquals(2, Node.POST_FLAG);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("5ZsHgkDg[q");
      Node node1 = compiler0.parseTestCode("5ZsHgkDg[q");
      Node node2 = new Node(108, node0, node0, node0, node1, 31, (-1));
      Node node3 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node2);
      assertEquals(0, Node.NON_SPECIALCALL);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
      Node node0 = new Node(98, 98, 98);
      Node node1 = new Node(98, node0, node0, node0, node0);
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
  public void test12()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
      Node node0 = new Node(100);
      Node node1 = new Node(113, node0, node0, node0, node0);
      node0.addChildrenToFront(node1);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
      assertEquals(4, Node.ENUM_PROP);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
      Node node0 = Node.newNumber((double) 113, 113, 113);
      node0.setDouble(0.0);
      Node node1 = new Node(113, node0, node0, node0, node0);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
      assertFalse(node2.isLocalResultCall());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
      Compiler compiler0 = new Compiler();
      FlowSensitiveInlineVariables flowSensitiveInlineVariables0 = new FlowSensitiveInlineVariables((AbstractCompiler) null);
      compiler0.setNormalized();
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, flowSensitiveInlineVariables0, (ScopeCreator) null);
      peepholeSubstituteAlternateSyntax0.beginTraversal(nodeTraversal0);
      Node node0 = Node.newString("8LA?Et.>[Z&'__4B3");
      Node node1 = new Node(30, node0, node0, node0, 43, 22);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
      assertTrue(node2.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
      Compiler compiler0 = new Compiler();
      ConstCheck constCheck0 = new ConstCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, constCheck0);
      peepholeSubstituteAlternateSyntax0.beginTraversal(nodeTraversal0);
      Node node0 = new Node(37, 37, 37);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertEquals(37, Node.SYNTHETIC_BLOCK_PROP);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true, true);
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      compiler0.setNormalized();
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, controlFlowAnalysis0, syntacticScopeCreator0);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax();
      peepholeSubstituteAlternateSyntax0.beginTraversal(nodeTraversal0);
      Node node0 = new Node(37, 37, 37);
      node0.addChildToBack(node0);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertEquals(9, Node.FIXUPS_PROP);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      boolean boolean0 = PeepholeSubstituteAlternateSyntax.containsUnicodeEscape("com.google.javascript.jscomp.PeepholeSubstituteAlternateSyntax$FoldArrayAction");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      String string0 = "\\u";
      boolean boolean0 = PeepholeSubstituteAlternateSyntax.containsUnicodeEscape(string0);
      assertTrue(boolean0);
  }
}
