/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:02:26 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.LightweightMessageFormatter;
import com.google.javascript.jscomp.LoggerErrorManager;
import com.google.javascript.jscomp.PeepholeSubstituteAlternateSyntax;
import com.google.javascript.rhino.Node;
import java.util.logging.Logger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PeepholeSubstituteAlternateSyntax_ESTest extends PeepholeSubstituteAlternateSyntax_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node0 = new Node(4, (-1975), 800);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertEquals(15, Node.NO_SIDE_EFFECTS);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node0 = Node.newString(" ms");
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertFalse(node1.isLabelName());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node0 = Node.newString("Ed#uTb^6upoDV1");
      Node node1 = new Node(43, node0, node0, node0, 32, 37);
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
  public void test03()  throws Throwable  {
      Node node0 = new Node(44, 83, 83);
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
  public void test04()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Node node0 = new Node(63, 63, (-926));
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertFalse(node1.isParamList());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Compiler compiler0 = new Compiler();
      peepholeSubstituteAlternateSyntax0.beginTraversal(compiler0);
      Node node0 = Node.newString("L_@@4");
      Node node1 = new Node(30, node0, node0, node0, 12, 43);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
      assertEquals(31, Node.SIDE_EFFECTS_FLAGS_MASK);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      Logger logger0 = Logger.getGlobal();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(lightweightMessageFormatter0, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      peepholeSubstituteAlternateSyntax0.beginTraversal(compiler0);
      Node node0 = Node.newString(" ms");
      Node node1 = new Node(37, node0, node0, node0, 7, 2);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      Logger logger0 = Logger.getGlobal();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(lightweightMessageFormatter0, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      peepholeSubstituteAlternateSyntax0.beginTraversal(compiler0);
      Node node0 = Node.newString(" ms");
      Node node1 = new Node(37, node0, node0, node0, 7, 2);
      node1.removeFirstChild();
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
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
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node0 = new Node(85, 85, 85);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertFalse(node1.isGetElem());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Node node0 = new Node(85, 85, 85);
      Node node1 = new Node(4, node0, node0, node0, 30, 4095);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertNotNull(node2);
      assertFalse(node2.hasChildren());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      Logger logger0 = Logger.getLogger("SAFE_TO_FOLD_WITHOUT_ARGS");
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(lightweightMessageFormatter0, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      peepholeSubstituteAlternateSyntax0.beginTraversal(compiler0);
      Node node0 = Node.newString("SAFE_TO_FOLD_WITHOUT_ARGS");
      Node node1 = new Node(38, node0, node0, node0, 32, 49);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
      assertFalse(node2.isFunction());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node0 = Node.newString("BITNOT");
      Node node1 = new Node(4, node0, node0, node0, 43, 47);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
      assertEquals(1, node2.getChildCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Node node0 = new Node(44, 83, 83);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertFalse(node1.isNE());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      Logger logger0 = Logger.getLogger("L_@@4");
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(lightweightMessageFormatter0, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Node node0 = compiler0.parseTestCode("L_@@4");
      Node node1 = new Node(63, node0, node0, node0, node0, 49, 43);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
      assertNull(node2.getSourceFileName());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node0 = Node.newString("Ed#uTb^6upoDV1");
      Node node1 = new Node(63, node0, node0, node0, 55, 39);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
      assertEquals(1, Node.DECR_FLAG);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      boolean boolean0 = PeepholeSubstituteAlternateSyntax.containsUnicodeEscape("?vdo`6p-k~T");
      assertTrue(boolean0);
  }
}