/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:44:18 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.CheckSideEffects;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CheckSideEffects_ESTest extends CheckSideEffects_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("SET");
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      checkSideEffects0.hotSwapScript(node0, node0);
      assertEquals(1, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CheckSideEffects.StripProtection checkSideEffects_StripProtection0 = new CheckSideEffects.StripProtection((AbstractCompiler) null);
      // Undeclared exception!
      try { 
        checkSideEffects_StripProtection0.process((Node) null, (Node) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ST");
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, false);
      checkSideEffects0.process(node0, node0);
      assertTrue(compiler0.hasErrors());
      assertEquals(1, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode(";T");
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      // Undeclared exception!
      try { 
        checkSideEffects0.process(node0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("SET");
      Node node1 = new Node(85, node0, 15, 31);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      Node node2 = new Node(37, node1, 4095, 30);
      // Undeclared exception!
      try { 
        checkSideEffects0.process(node0, node2);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("!!h,JIt8");
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      // Undeclared exception!
      try { 
        checkSideEffects0.process(node0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString("f~/bn1_fu>f");
      Node node1 = new Node(85, node0, 15, 31);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      Node node2 = new Node(37, node1, 4095, 30);
      Node node3 = Node.newNumber((double) 4095);
      node2.addChildrenToFront(node3);
      checkSideEffects0.process(node2, node1);
      assertEquals(50, Node.FREE_CALL);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("KH'{=gxCE");
      Node node1 = new Node(85, node0, 15, 31);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      Node node2 = new Node(37, node1, 4095, 30);
      // Undeclared exception!
      try { 
        checkSideEffects0.visit((NodeTraversal) null, node2, node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString("KH>{=gxCE");
      Node node1 = new Node(85, node0, 15, 31);
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      Node node2 = new Node(85, node1, 41, 54);
      // Undeclared exception!
      try { 
        checkSideEffects0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("KH'{=gxCE");
      Node node1 = new Node(85, node0, 15, 31);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      Node node2 = new Node(125, node1, 37, 32);
      checkSideEffects0.process(node1, node1);
      assertFalse(node1.isWith());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("KH'{=gxCE");
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      Node node1 = new Node(125, node0, 2, 45);
      checkSideEffects0.process(node0, node1);
      assertEquals(0, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ST");
      CheckSideEffects.StripProtection checkSideEffects_StripProtection0 = new CheckSideEffects.StripProtection(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkSideEffects_StripProtection0);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, false);
      Node node1 = new Node(115, node0, 4, 16);
      checkSideEffects0.visit(nodeTraversal0, node0, node1);
      assertFalse(node1.isDo());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, false);
      Node node0 = Node.newString(130, "", 130, 86);
      Node node1 = new Node(0, node0, 53, (-1995));
      checkSideEffects0.visit((NodeTraversal) null, node0, node0);
      assertFalse(node0.isDefaultCase());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ST");
      Node node1 = new Node(85, node0, 15, (-1063));
      CheckSideEffects.StripProtection checkSideEffects_StripProtection0 = new CheckSideEffects.StripProtection(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkSideEffects_StripProtection0);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, false);
      Node node2 = new Node(115, node1, 4, 16);
      Node node3 = Node.newString(130, "ST", 53, 41);
      node1.addChildrenToFront(node3);
      checkSideEffects0.visit(nodeTraversal0, node3, node3);
      assertEquals(47, Node.IS_DISPATCHER);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("KH'{=gxCE");
      Node node1 = new Node(29, node0, 4095, 16);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      Node node2 = new Node(125, node1, 2, 45);
      checkSideEffects0.process(node0, node2);
      assertEquals(2, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CheckSideEffects.StripProtection checkSideEffects_StripProtection0 = new CheckSideEffects.StripProtection((AbstractCompiler) null);
      Node node0 = Node.newString("t_=I48*.");
      checkSideEffects_StripProtection0.visit((NodeTraversal) null, node0, node0);
      assertEquals(12, Node.COLUMN_BITS);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("$m^W2gF=^h");
      Node node1 = new Node(37, node0, 4095, 30);
      CheckSideEffects.StripProtection checkSideEffects_StripProtection0 = new CheckSideEffects.StripProtection(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkSideEffects_StripProtection0);
      checkSideEffects_StripProtection0.visit(nodeTraversal0, node1, node1);
      assertEquals(39, Node.EMPTY_BLOCK);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("$m^W2gF=^h");
      Node node1 = new Node(38, node0, 15, 31);
      Node node2 = new Node(37, node1, 4095, 30);
      CheckSideEffects.StripProtection checkSideEffects_StripProtection0 = new CheckSideEffects.StripProtection(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkSideEffects_StripProtection0);
      // Undeclared exception!
      try { 
        checkSideEffects_StripProtection0.visit(nodeTraversal0, node2, node2);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // NAME 15 is not a string node
         //
         verifyException("com.google.javascript.rhino.Node", e);
      }
  }
}
