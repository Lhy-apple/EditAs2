/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:18:40 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ControlFlowAnalysis;
import com.google.javascript.jscomp.ControlFlowGraph;
import com.google.javascript.jscomp.JsAst;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.jscomp.SourceFile;
import com.google.javascript.jscomp.TightenTypes;
import com.google.javascript.jscomp.VarCheck;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.io.PrintStream;
import java.util.List;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ControlFlowAnalysis_ESTest extends ControlFlowAnalysis_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$1", "com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$1");
      Node node1 = new Node(1, node0, (-823), 12);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      controlFlowAnalysis0.process(node1, node1);
      assertEquals(42, Node.NO_SIDE_EFFECTS_CALL);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Node node0 = Node.newString(115, "com.google.common.base.Objects");
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node1 = new Node(49, node0, 431, 1095);
      controlFlowAnalysis0.process(node0, node1);
      assertEquals(4, Node.DESCENDANTS_FLAG);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Node node0 = Node.newString(112, "", 112, 112);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      controlFlowAnalysis0.process(node0, node0);
      assertFalse(node0.wasEmptyNode());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Node node0 = new Node(113);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      ControlFlowGraph<Node> controlFlowGraph0 = controlFlowAnalysis0.getCfg();
      assertNull(controlFlowGraph0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Node node0 = Node.newString(74, "i:G#FGUkG50._E|");
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node1 = new Node(114, node0, node0);
      controlFlowAnalysis0.process(node0, node1);
      assertEquals(42, Node.NO_SIDE_EFFECTS_CALL);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(105, "<(7 PBY t'V8a5=", 44, 44);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
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
  public void test07()  throws Throwable  {
      Node node0 = Node.newString(105, "com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$1");
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
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
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(4, "#8N17{Xapm ~l", (-966), (-966));
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      Node node1 = new Node(4, node0, 1670, 23);
      controlFlowAnalysis0.process(node1, node1);
      assertFalse(node1.hasMoreThanOneChild());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(77, "com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$1");
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      Node node1 = Node.newNumber((-1004.3298182213737));
      node0.addChildToBack(node1);
      controlFlowAnalysis0.process(node1, node0);
      assertFalse(node0.isSyntheticBlock());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = Node.newNumber((double) 142, 142, 142);
      Node node1 = Node.newString(105, "#8N17{Xapm ~l", 2, 30);
      controlFlowAnalysis0.process(node0, node0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, controlFlowAnalysis0, (ScopeCreator) null);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.shouldTraverse(nodeTraversal0, node0, node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ControlFlowAnalysis", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newNumber((double) 119, 119, 119);
      Node node1 = Node.newString(108, "#8N17{Xapm ~l", 4095, 34);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      controlFlowAnalysis0.process(node1, node0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, controlFlowAnalysis0, (ScopeCreator) null);
      boolean boolean0 = controlFlowAnalysis0.shouldTraverse(nodeTraversal0, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(110);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      node0.addChildToBack(node0);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Node node0 = Node.newString(111, "|M@o(", 111, 111);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      Node node1 = new Node(111, node0, 4, (-3));
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = Node.newNumber((double) 142, 142, 142);
      Node node1 = Node.newString(116, "#8N17{Xapm ~l", 2, 30);
      controlFlowAnalysis0.process(node0, node0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, controlFlowAnalysis0, (ScopeCreator) null);
      boolean boolean0 = controlFlowAnalysis0.shouldTraverse(nodeTraversal0, node0, node1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Node node0 = new Node(119);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node1 = new Node(117, node0, node0, 13, 27);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(119);
      Node node1 = Node.newString(21, "p=hi}XG3,G{", 23, 0);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      controlFlowAnalysis0.process(node1, node1);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, controlFlowAnalysis0);
      boolean boolean0 = controlFlowAnalysis0.shouldTraverse(nodeTraversal0, node0, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(129);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      Node node1 = compiler0.parseSyntheticCode("CvrT:++p9L+W", "CvrT:++p9L+W");
      controlFlowAnalysis0.process(node0, node1);
      assertEquals(1, Node.TARGET_PROP);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(74, "com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$1");
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      Node node1 = new Node(115, node0, 12, (-3));
      controlFlowAnalysis0.process(node1, node0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, controlFlowAnalysis0);
      boolean boolean0 = controlFlowAnalysis0.shouldTraverse(nodeTraversal0, node1, node1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Node node0 = new Node(119);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node1 = new Node(119, node0, node0, 13, 27);
      controlFlowAnalysis0.process(node0, node1);
      assertEquals((-3), Node.LOCAL_BLOCK_PROP);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(105, "<(7 PBY t'V8a5=", 44, 44);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, controlFlowAnalysis0);
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
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newNumber((double) 119, 119, 119);
      Node node1 = Node.newString(108, "#8N17{Xapm ~l", 4095, 34);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      VarCheck varCheck0 = new VarCheck(compiler0);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = new Node(115);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0);
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
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newNumber((double) 119, 119, 119);
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded((String) null, "#8N1(y7{Mapm ~l");
      JsAst jsAst0 = new JsAst(sourceFile_Preloaded0);
      Node node1 = jsAst0.getAstRoot(compiler0);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      controlFlowAnalysis0.process(node0, node1);
      assertEquals(2, Node.RIGHT);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newNumber((double) 119, 119, 119);
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded((String) null, "#8N1(y7{Mapm ~l");
      JsAst jsAst0 = new JsAst(sourceFile_Preloaded0);
      Node node1 = jsAst0.getAstRoot(compiler0);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node2 = new Node(9, node1, node0);
      controlFlowAnalysis0.process(node0, node2);
      assertEquals(35, Node.QUOTED_PROP);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis((AbstractCompiler) null, false);
      Node node0 = new Node(112);
      Node[] nodeArray0 = new Node[1];
      nodeArray0[0] = node0;
      Node node1 = new Node(116, nodeArray0);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.visit((NodeTraversal) null, node1, node0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // DEFAULT is not a string node
         //
         verifyException("com.google.javascript.rhino.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = new Node(116);
      Node node1 = new Node(40, node0, 37, 31);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node1, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // Cannot find break target.
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(117, "#8N1(y7{Mapm ~l", 14, 0);
      Node node1 = new Node(23, node0, 132, 3);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
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
  public void test28()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(4, "#8N17{Xapm ~l", (-966), (-966));
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      controlFlowAnalysis0.process(node0, node0);
      assertEquals(26, Node.DIRECTCALL_PROP);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Node node0 = Node.newString(85, "TO?");
      Compiler compiler0 = new Compiler((PrintStream) null);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node1 = new Node(105, node0, node0, node0, node0);
      controlFlowAnalysis0.process(node0, node0);
      assertEquals(26, Node.DIRECTCALL_PROP);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      VarCheck varCheck0 = new VarCheck(compiler0);
      Node node0 = Node.newNumber(0.0, 119, 119);
      Node node1 = Node.newString(108, "#8N17{Xapm ~l", 4095, 34);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0, (ScopeCreator) null);
      node1.addChildToBack(node0);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.visit(nodeTraversal0, node0, node1);
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
      Node node0 = Node.newString(111, "|M@o(", 111, 111);
      Compiler compiler0 = new Compiler();
      Node node1 = Node.newString("~as/", (-20), 112);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      node0.addChildToBack(node1);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, controlFlowAnalysis0);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.visit(nodeTraversal0, node1, node1);
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
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis((AbstractCompiler) null, false);
      Node node0 = new Node(112);
      Stack<JSType> stack0 = new Stack<JSType>();
      Compiler compiler0 = new Compiler();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      Node node1 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) stack0);
      node0.addChildToBack(node1);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, controlFlowAnalysis0);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.visit(nodeTraversal0, node1, node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ControlFlowAnalysis", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Node node0 = Node.newString(74, "com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$1");
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      Node node1 = new Node(115, node0, 12, (-3));
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(77, "com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$1");
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      Node node1 = new Node(30, node0, 2, 32);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(35);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      controlFlowAnalysis0.process(node0, node0);
      assertEquals(5, Node.FUNCTION_PROP);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Node node0 = Node.newString(37, "6)sT#");
      MockPrintStream mockPrintStream0 = new MockPrintStream("Cannot find continue target.");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      controlFlowAnalysis0.process(node0, node0);
      assertEquals(48, Node.DIRECTIVES);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(103);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      controlFlowAnalysis0.process(node0, node0);
      assertEquals(22, Node.TARGETBLOCK_PROP);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(105, "<(7 PBY t'V8a5=", 44, 44);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node1 = new Node(160, node0, node0, node0);
      controlFlowAnalysis0.process(node0, node1);
      assertEquals(12, Node.COLUMN_BITS);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Node node0 = Node.newString(108, "#8N17{Xapm ~l", 4095, 34);
      boolean boolean0 = ControlFlowAnalysis.isBreakStructure(node0, false);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Node node0 = new Node(110);
      boolean boolean0 = ControlFlowAnalysis.isBreakStructure(node0, false);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Node node0 = Node.newString(114, "", 114, 114);
      boolean boolean0 = ControlFlowAnalysis.isBreakStructure(node0, false);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Node node0 = new Node(115);
      boolean boolean0 = ControlFlowAnalysis.isBreakStructure(node0, true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Node node0 = new Node(125);
      boolean boolean0 = ControlFlowAnalysis.isBreakStructure(node0, false);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Node node0 = Node.newNumber(0.0, 119, 119);
      Node node1 = new Node(114, node0, node0, node0);
      boolean boolean0 = ControlFlowAnalysis.isContinueStructure(node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Node node0 = new Node(115);
      boolean boolean0 = ControlFlowAnalysis.isContinueStructure(node0);
      assertTrue(boolean0);
  }
}