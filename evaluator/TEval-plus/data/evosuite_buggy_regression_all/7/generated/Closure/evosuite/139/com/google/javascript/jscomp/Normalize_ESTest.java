/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:21:34 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Normalize_ESTest extends Normalize_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize normalize0 = new Normalize(compiler0, false);
      Node node0 = compiler0.parseSyntheticCode("E", "E");
      Node node1 = new Node(37, node0, node0, node0, node0);
      normalize0.process(node0, node0);
      assertTrue(node0.isSyntheticBlock());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants((AbstractCompiler) null, true);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, normalize_VerifyConstants0);
      Node node0 = Node.newNumber(0.0);
      normalize_VerifyConstants0.visit(nodeTraversal0, node0, node0);
      assertEquals(0, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, false);
      Node node0 = Node.newString(3296, "");
      // Undeclared exception!
      try { 
        normalize_VerifyConstants0.process(node0, node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(105);
      node0.addChildrenToFront(node0);
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, true);
      // Undeclared exception!
      try { 
        normalize_VerifyConstants0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants((AbstractCompiler) null, true);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, normalize_VerifyConstants0);
      Node node0 = Node.newNumber(0.0);
      Node node1 = new Node(38, node0, node0);
      // Undeclared exception!
      try { 
        normalize_VerifyConstants0.visit(nodeTraversal0, node1, node0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // NAME is not a string node
         //
         verifyException("com.google.javascript.rhino.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(126);
      Node node1 = new Node(118, node0, node0);
      node0.addChildrenToFront(node1);
      Normalize normalize0 = new Normalize(compiler0, false);
      // Undeclared exception!
      try { 
        normalize0.process(node1, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(126);
      Node node1 = new Node(123, node0, node0);
      node0.addChildrenToFront(node1);
      Normalize normalize0 = new Normalize(compiler0, true);
      // Undeclared exception!
      try { 
        normalize0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // Normalize constraints violated:
         // LABEL normalization
         //
         verifyException("com.google.javascript.jscomp.Normalize$NormalizeStatements", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Node node0 = new Node(113);
      Normalize normalize0 = new Normalize((AbstractCompiler) null, false);
      // Undeclared exception!
      try { 
        normalize0.process(node0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(126);
      Node node1 = new Node(114, node0, node0);
      node0.addChildrenToFront(node1);
      Normalize normalize0 = new Normalize(compiler0, true);
      // Undeclared exception!
      try { 
        normalize0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(126);
      node0.addChildrenToFront(node0);
      Normalize normalize0 = new Normalize(compiler0, true);
      // Undeclared exception!
      try { 
        normalize0.process(node0, node0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(126);
      Node node1 = new Node(48, node0, node0);
      node0.addChildrenToFront(node1);
      Normalize normalize0 = new Normalize(compiler0, false);
      // Undeclared exception!
      try { 
        normalize0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(105);
      node0.addChildrenToFront(node0);
      Normalize normalize0 = new Normalize(compiler0, true);
      // Undeclared exception!
      try { 
        normalize0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Node node0 = new Node(126, 126, 126);
      Node node1 = new Node(105, node0, node0);
      node0.addChildrenToFront(node1);
      Normalize normalize0 = new Normalize((AbstractCompiler) null, true);
      // Undeclared exception!
      try { 
        normalize0.process(node1, node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }
}