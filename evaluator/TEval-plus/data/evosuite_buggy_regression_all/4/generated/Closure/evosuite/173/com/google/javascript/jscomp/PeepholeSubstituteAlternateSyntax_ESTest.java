/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:27:15 GMT 2023
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
      Node node0 = new Node(4, 4, 4);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertTrue(node1.isReturn());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Node node0 = Node.newString("]L[#&;");
      Node node1 = new Node(43, node0, node0, node0, node0);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
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
  public void test02()  throws Throwable  {
      Node node0 = new Node(44, 44, (-567));
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
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Node node0 = Node.newString("jQuery.expandedEach was not expanded as no valid property assignments were encountered. Consider using jQuery.each intead.");
      Node node1 = new Node(63, node0, node0, node0, node0);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
      assertFalse(node2.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Node node0 = Node.newString("]L[#&;");
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertFalse(node1.isNumber());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Node node0 = new Node(30);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Compiler compiler0 = new Compiler();
      peepholeSubstituteAlternateSyntax0.beginTraversal(compiler0);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertFalse(node1.isFor());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Node node0 = Node.newString(37, "gvtWsu}rnWWOq{^o]2", 514, 514);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      Logger logger0 = Logger.getGlobal();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(lightweightMessageFormatter0, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      peepholeSubstituteAlternateSyntax0.beginTraversal(compiler0);
      // Undeclared exception!
      try { 
        peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
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
      Node node0 = Node.newString(37, "gvtWsu}rnWWOq{^o]2", 514, 514);
      Node node1 = new Node(37, node0, node0, node0, node0);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      Logger logger0 = Logger.getGlobal();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(lightweightMessageFormatter0, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      peepholeSubstituteAlternateSyntax0.beginTraversal(compiler0);
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
      Node node0 = Node.newString(85, "jQuery.expandedEach was not expanded as no valid property assignments were encountered. Consider using jQuery.each instead.");
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertEquals(30, Node.VAR_ARGS_NAME);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Node node0 = Node.newString(85, "jQuery.expandedEach was not expanded as no valid property assignments were encountered. Consider using jQuery.each instead.");
      Node node1 = new Node(30, node0, node0, 16, 55);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertNotNull(node2);
      assertFalse(node2.isNew());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Node node0 = new Node(38);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Compiler compiler0 = new Compiler();
      peepholeSubstituteAlternateSyntax0.beginTraversal(compiler0);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertFalse(node1.isThis());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Node node0 = Node.newString("");
      Node node1 = new Node(4, node0, node0, node0, node0);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
      assertEquals(1, node2.getChildCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(false);
      Node node0 = Node.newString(44, "m4in");
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertEquals(12, Node.COLUMN_BITS);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Node node0 = Node.newString(63, "", 63, 63);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      node0.addChildToBack(node0);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertEquals(0, Node.SIDE_EFFECTS_ALL);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Node node0 = new Node(63);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node1 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node0);
      assertFalse(node1.isCall());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      boolean boolean0 = PeepholeSubstituteAlternateSyntax.containsUnicodeEscape("jQuery.expandedEach was not expanded as no valid property assignments were encountered. Consider using jQuery.each intead.");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      boolean boolean0 = PeepholeSubstituteAlternateSyntax.containsUnicodeEscape("k/r{pGL1&(|>y");
      assertTrue(boolean0);
  }
}