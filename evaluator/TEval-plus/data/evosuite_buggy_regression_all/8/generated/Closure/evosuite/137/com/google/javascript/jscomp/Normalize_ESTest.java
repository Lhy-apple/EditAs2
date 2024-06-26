/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:29:13 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Normalize_ESTest extends Normalize_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode(" df", " df");
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
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, true);
      Node node0 = Node.newString("() ");
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
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize.PropogateConstantAnnotations normalize_PropogateConstantAnnotations0 = new Normalize.PropogateConstantAnnotations(compiler0, false);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, normalize_PropogateConstantAnnotations0);
      Node node0 = Node.newString("UDy.BS@jk", 4996, 4996);
      normalize_PropogateConstantAnnotations0.visit(nodeTraversal0, node0, node0);
      assertEquals(1, Node.DECR_FLAG);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize.PropogateConstantAnnotations normalize_PropogateConstantAnnotations0 = new Normalize.PropogateConstantAnnotations(compiler0, true);
      // Undeclared exception!
      try { 
        normalize_PropogateConstantAnnotations0.process((Node) null, (Node) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Normalize.PropogateConstantAnnotations normalize_PropogateConstantAnnotations0 = new Normalize.PropogateConstantAnnotations((AbstractCompiler) null, false);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, normalize_PropogateConstantAnnotations0);
      Node node0 = Node.newString(38, "#", 38, 38);
      // Undeclared exception!
      try { 
        normalize_PropogateConstantAnnotations0.visit(nodeTraversal0, node0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Normalize$PropogateConstantAnnotations", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize.PropogateConstantAnnotations normalize_PropogateConstantAnnotations0 = new Normalize.PropogateConstantAnnotations(compiler0, false);
      Node node0 = Node.newString(38, "", 38, 1052);
      NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) normalize_PropogateConstantAnnotations0);
      assertEquals(40, Node.ORIGINALNAME_PROP);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, false);
      Node node0 = Node.newString("");
      Node node1 = new Node(124, node0, node0, node0, node0);
      // Undeclared exception!
      try { 
        normalize_VerifyConstants0.process(node1, node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants((AbstractCompiler) null, true);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, normalize_VerifyConstants0);
      Node node0 = new Node(86);
      normalize_VerifyConstants0.visit(nodeTraversal0, node0, node0);
      assertEquals(4, Node.ENUM_PROP);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, true);
      Node node0 = Node.newString(38, "L4$Kvd(M=^~3'", 38, 38);
      NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) normalize_VerifyConstants0);
      NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) normalize_VerifyConstants0);
      assertFalse(node0.isSyntheticBlock());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, true);
      Node node0 = Node.newString(38, "", (-547), (-3557));
      NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) normalize_VerifyConstants0);
      assertEquals(41, Node.BRACELESS_TYPE);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants((AbstractCompiler) null, false);
      Node node0 = Node.newString(38, "L4$Kvd(M=^~3'", 83, 38);
      NodeTraversal.traverse((AbstractCompiler) null, node0, (NodeTraversal.Callback) normalize_VerifyConstants0);
      assertEquals(39, Node.EMPTY_BLOCK);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, true);
      Node node0 = Node.newString(38, "Q)C\"", 38, 38);
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) normalize_VerifyConstants0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // The name Q)C\" is not annotated as constant.
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize.NormalizeStatements normalize_NormalizeStatements0 = new Normalize.NormalizeStatements(compiler0, false);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, normalize_NormalizeStatements0, typedScopeCreator0);
      Node node0 = Node.newString(113, "L4$Kvd(M=^~3'", 1542, 1542);
      Node node1 = new Node(113, node0, node0, node0, node0, 0, 28);
      normalize_NormalizeStatements0.visit(nodeTraversal0, node1, node0);
      assertEquals(3, node1.getChildCount());
      assertEquals(113, node0.getType());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize.NormalizeStatements normalize_NormalizeStatements0 = new Normalize.NormalizeStatements(compiler0, true);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, normalize_NormalizeStatements0, typedScopeCreator0);
      Node node0 = Node.newString(113, "L4$Kvd(M=~3'", 1542, 1542);
      Node node1 = new Node(113, node0, node0, node0, node0, 0, 28);
      // Undeclared exception!
      try { 
        normalize_NormalizeStatements0.visit(nodeTraversal0, node1, node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Normalize constraints violated:
         // WHILE node
         //
         verifyException("com.google.javascript.jscomp.Normalize$NormalizeStatements", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(126, "2t1k48203Sb[mBAc:w", 3470, 3470);
      Normalize normalize0 = new Normalize(compiler0, true);
      // Undeclared exception!
      try { 
        normalize0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(105, "]", 105, (-3547));
      Normalize normalize0 = new Normalize(compiler0, true);
      // Undeclared exception!
      try { 
        normalize0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }
}
