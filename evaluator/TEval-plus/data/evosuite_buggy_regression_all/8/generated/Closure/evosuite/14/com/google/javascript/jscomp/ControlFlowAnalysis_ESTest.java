/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:04:21 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ControlFlowAnalysis;
import com.google.javascript.jscomp.ControlFlowGraph;
import com.google.javascript.jscomp.LoggerErrorManager;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.SyntheticAst;
import com.google.javascript.jscomp.Tracer;
import com.google.javascript.rhino.Node;
import java.util.logging.Logger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ControlFlowAnalysis_ESTest extends ControlFlowAnalysis_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Node node0 = new Node((-468));
      Node node1 = new Node(119, node0, node0);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true, false);
      controlFlowAnalysis0.process(node0, node1);
      assertFalse(node1.isIf());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SyntheticAst syntheticAst0 = new SyntheticAst("h y.06(!A?\"cJG");
      Logger logger0 = Tracer.logger;
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Node node0 = syntheticAst0.getAstRoot(compiler0);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      Node node1 = new Node(77, node0, (-704), (-704));
      controlFlowAnalysis0.process(node0, node1);
      assertFalse(node0.isDo());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Node node0 = Node.newNumber((-9.426936519809457));
      Node node1 = new Node(49, node0, node0, node0, node0);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true, true);
      controlFlowAnalysis0.process(node0, node1);
      assertFalse(node1.isFunction());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SyntheticAst syntheticAst0 = new SyntheticAst("ObrZ*LzuT5");
      Logger logger0 = Tracer.logger;
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Node node0 = syntheticAst0.getAstRoot(compiler0);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true, false);
      Node node1 = new Node(112, node0, 12, 45);
      controlFlowAnalysis0.process(node0, node1);
      assertFalse(node0.isFor());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ObrZ*LzuT");
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true, true);
      Node node1 = new Node(113, node0, 126, 39);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis((AbstractCompiler) null, true, true);
      ControlFlowGraph<Node> controlFlowGraph0 = controlFlowAnalysis0.getCfg();
      assertNull(controlFlowGraph0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SyntheticAst syntheticAst0 = new SyntheticAst("(>U'p%)%;ViK");
      Node node0 = syntheticAst0.getAstRoot(compiler0);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true, true);
      Node node1 = new Node(114, node0, 45, 52);
      controlFlowAnalysis0.process(node1, node1);
      assertEquals(4095, Node.MAX_COLUMN_NUMBER);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SyntheticAst syntheticAst0 = new SyntheticAst("(>U'p%)%;ViK");
      Node node0 = syntheticAst0.getAstRoot(compiler0);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true, true);
      Node node1 = new Node(120, node0, 45, 52);
      controlFlowAnalysis0.process(node1, node1);
      assertFalse(node1.isGetProp());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Node node0 = new Node(105, 105, 105);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node0);
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
  public void test09()  throws Throwable  {
      Node node0 = new Node(105);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis((AbstractCompiler) null, true, true);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Node node0 = new Node(105, 105, 105);
      Node node1 = new Node(43, node0, node0, node0, node0);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      controlFlowAnalysis0.process(node0, node1);
      assertFalse(node1.wasEmptyNode());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Node node0 = Node.newNumber((double) 133);
      Node node1 = new Node(4, node0, node0, node0, node0);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      controlFlowAnalysis0.process(node0, node1);
      assertFalse(node0.isNew());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Node node0 = new Node((byte)108);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true, true);
      Node node1 = new Node((byte)108, node0, 45, 120);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ObrZ*LzuT5");
      Node node1 = new Node(110, node0, node0);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      controlFlowAnalysis0.process(node1, node1);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, controlFlowAnalysis0);
      boolean boolean0 = controlFlowAnalysis0.shouldTraverse(nodeTraversal0, node1, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(139, 139, 139);
      Node node1 = new Node(111, node0, 29, 53);
      Node node2 = new Node(2, node1, 2, 53);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      controlFlowAnalysis0.process(node2, node2);
      assertFalse(node2.isBreak());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Node node0 = Node.newNumber((-2.0290263605130976));
      Node node1 = new Node(115, node0, node0, node0, node0);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Node node0 = new Node(41);
      Node node1 = new Node(116, node0, node0);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newNumber((-457.7));
      Node node1 = new Node(117, node0, node0, 41, 12);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Node node0 = new Node(118, 118, 118);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis((AbstractCompiler) null, true, false);
      Node node1 = new Node(118, node0, 38, 46);
      controlFlowAnalysis0.process(node1, node1);
      assertEquals(29, Node.JSDOC_INFO_PROP);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SyntheticAst syntheticAst0 = new SyntheticAst("h y.06(!A?\"cJG");
      Logger logger0 = Tracer.logger;
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Node node0 = syntheticAst0.getAstRoot(compiler0);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      Node node1 = new Node(126, node0, 36, 52);
      controlFlowAnalysis0.process(node0, node1);
      assertFalse(node1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Node node0 = new Node((-468));
      Node node1 = new Node(119, node0, node0);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true, true);
      controlFlowAnalysis0.process(node1, node0);
      boolean boolean0 = controlFlowAnalysis0.shouldTraverse((NodeTraversal) null, node1, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Node node0 = new Node(105, 105, 105);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis((AbstractCompiler) null, true, true);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, controlFlowAnalysis0);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.visit(nodeTraversal0, node0, node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Node node0 = new Node(115);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis((AbstractCompiler) null, false, false);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.visit((NodeTraversal) null, node0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ControlFlowAnalysis", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Node node0 = new Node(125);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      controlFlowAnalysis0.process(node0, node0);
      assertFalse(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ObrZ*LzuT");
      Node node1 = new Node(111, node0, 1143, 1);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true, false);
      controlFlowAnalysis0.process(node1, node0);
      assertEquals(38, Node.SYNTHETIC_BLOCK_PROP);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Node node0 = new Node(116, 116, (-1244));
      Node node1 = new Node(37, node0, node0);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Node node0 = new Node(117, 117, (byte)86);
      Compiler compiler0 = new Compiler();
      Node node1 = new Node(2, node0, 16, 44);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true, false);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node1, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // Cannot find continue target.
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Node node0 = new Node(113, 113, 113);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true, true);
      Node node1 = Node.newString(4, "com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$2", 45, 102);
      controlFlowAnalysis0.process(node0, node1);
      assertFalse(node1.isOr());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Node node0 = new Node(86);
      Node node1 = new Node(105, node0, node0, node0, node0);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      controlFlowAnalysis0.process(node0, node0);
      assertFalse(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Node node0 = Node.newNumber((-2.0290263605130976));
      Node node1 = new Node(115, node0, node0, node0, node0);
      // Undeclared exception!
      try { 
        ControlFlowAnalysis.computeFollowNode(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ControlFlowAnalysis", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      SyntheticAst syntheticAst0 = new SyntheticAst("ObrZ*LzuT5");
      Logger logger0 = Tracer.logger;
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Node node0 = syntheticAst0.getAstRoot(compiler0);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      Node node1 = new Node(108, node0, node0);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.visit((NodeTraversal) null, node0, node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ControlFlowAnalysis", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(150, 150, 150);
      Node node1 = new Node(111, node0, 29, 53);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, controlFlowAnalysis0);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.visit(nodeTraversal0, node0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ControlFlowAnalysis", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      SyntheticAst syntheticAst0 = new SyntheticAst("<*o");
      Logger logger0 = Tracer.logger;
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Node node0 = syntheticAst0.getAstRoot(compiler0);
      Node node1 = new Node(113, node0, node0);
      Node node2 = ControlFlowAnalysis.computeFollowNode(node0);
      assertNotNull(node2);
      assertEquals(113, node2.getType());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Node node0 = new Node(105);
      Node node1 = Node.newNumber((double) 8);
      Node node2 = new Node(42, node0, node1, node0, node0);
      Node node3 = ControlFlowAnalysis.computeFollowNode(node1);
      assertNull(node3);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Node node0 = Node.newNumber(32.0);
      Node node1 = Node.newNumber((double) 43);
      Node node2 = new Node(40, node0, node0, node1, node1);
      Node node3 = ControlFlowAnalysis.computeFollowNode(node0);
      assertFalse(node3.isOnlyModifiesThisCall());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Node node0 = new Node(86);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis((AbstractCompiler) null, true, false);
      Node node1 = new Node(77, node0, 38, 46);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Node node0 = new Node(105, 105, 105);
      boolean boolean0 = ControlFlowAnalysis.isBreakTarget(node0, "L\"a-b:7$");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Node node0 = Node.newNumber((-2.0290263605130976));
      Node node1 = new Node(115, node0, node0, node0, node0);
      boolean boolean0 = ControlFlowAnalysis.isBreakTarget(node1, (String) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Node node0 = new Node(113, 113, 113);
      Node node1 = new Node(43, node0, node0, node0, node0);
      boolean boolean0 = ControlFlowAnalysis.isBreakTarget(node0, "L\"a-b:7$");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Node node0 = new Node(133);
      Node node1 = new Node(30, node0, node0);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true, true);
      controlFlowAnalysis0.process(node0, node1);
      assertFalse(node1.equals((Object)node0));
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.ControlFlowAnalysis");
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, true);
      controlFlowAnalysis0.process(node0, node0);
      assertEquals(8, Node.FLAG_NO_THROWS);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Node node0 = new Node(35, 2, 41);
      boolean boolean0 = ControlFlowAnalysis.mayThrowException(node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Node node0 = new Node(125, 125, (-1244));
      Node node1 = new Node(37, node0, node0);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false, false);
      controlFlowAnalysis0.process(node0, node1);
      assertFalse(node1.isName());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Node node0 = Node.newString("a>*P(");
      Node node1 = new Node(52, node0, node0, node0, 2, 43);
      boolean boolean0 = ControlFlowAnalysis.mayThrowException(node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Node node0 = new Node(102);
      boolean boolean0 = ControlFlowAnalysis.mayThrowException(node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Node node0 = new Node(103);
      boolean boolean0 = ControlFlowAnalysis.mayThrowException(node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Node node0 = new Node(77);
      boolean boolean0 = ControlFlowAnalysis.isBreakStructure(node0, true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Node node0 = new Node((byte)108);
      // Undeclared exception!
      try { 
        ControlFlowAnalysis.isBreakTarget(node0, "com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$2");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ControlFlowAnalysis", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Node node0 = new Node(110);
      boolean boolean0 = ControlFlowAnalysis.isBreakStructure(node0, false);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SyntheticAst syntheticAst0 = new SyntheticAst("com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$2");
      Node node0 = syntheticAst0.getAstRoot(compiler0);
      Node node1 = new Node(114, node0, 45, 52);
      // Undeclared exception!
      try { 
        ControlFlowAnalysis.isBreakTarget(node1, "com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$2");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ControlFlowAnalysis", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Node node0 = new Node(125, 125, 125);
      boolean boolean0 = ControlFlowAnalysis.isBreakTarget(node0, (String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Node node0 = new Node(113);
      boolean boolean0 = ControlFlowAnalysis.isContinueStructure(node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Node node0 = Node.newString(114, "");
      boolean boolean0 = ControlFlowAnalysis.isContinueStructure(node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Node node0 = new Node(115);
      boolean boolean0 = ControlFlowAnalysis.isContinueStructure(node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ObrZ*LfzuT5");
      Node node1 = ControlFlowAnalysis.getExceptionHandler(node0);
      assertNull(node1);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Node node0 = new Node(105);
      Node node1 = ControlFlowAnalysis.getExceptionHandler(node0);
      assertNull(node1);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Node node0 = new Node(125);
      Node node1 = new Node(45, node0, node0, node0, 4, 42);
      // Undeclared exception!
      try { 
        ControlFlowAnalysis.getExceptionHandler(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ControlFlowAnalysis", e);
      }
  }
}
