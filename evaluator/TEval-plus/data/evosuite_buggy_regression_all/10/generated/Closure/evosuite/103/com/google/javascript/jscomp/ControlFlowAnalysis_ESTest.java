/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:47:38 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ControlFlowAnalysis;
import com.google.javascript.jscomp.ControlFlowGraph;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.OptimizeArgumentsArray;
import com.google.javascript.jscomp.SymbolTable;
import com.google.javascript.jscomp.SyntacticScopeCreator;
import com.google.javascript.jscomp.SyntheticAst;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.ScriptOrFnNode;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ControlFlowAnalysis_ESTest extends ControlFlowAnalysis_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.ControlFlowAnalysis$1", "com.google.javascript.jscomp.ControlFlowAnalysis$1");
      Node node1 = new Node(829);
      Node node2 = new Node(77, node1, node1, node1, node1, (-118), 84);
      controlFlowAnalysis0.process(node0, node2);
      assertFalse(node2.wasEmptyNode());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = Node.newString("");
      Node node1 = new Node(112, node0);
      controlFlowAnalysis0.process(node0, node1);
      assertEquals(1, node1.getChildCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = Node.newString(127, "com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$2");
      Node node1 = new Node(113, node0, node0, node0, 48, 2);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      ControlFlowGraph<Node> controlFlowGraph0 = controlFlowAnalysis0.getCfg();
      assertNull(controlFlowGraph0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = Node.newString(114, "com.google.javascript.jscomp.ControlFlowAnalysis");
      Node node1 = new Node((-1574), node0, node0, node0);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(105, 2624, 2624);
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
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      Node node0 = new Node(105, 2618, 2618);
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
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(105, 2624, 2624);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node1 = new Node(108, node0);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      Node node0 = Node.newString("<unknown=-822>");
      Node node1 = new Node(4, node0);
      controlFlowAnalysis0.process(node0, node1);
      assertEquals(29, Node.VAR_ARGS_NAME);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SymbolTable symbolTable0 = compiler0.acquireSymbolTable();
      Node node0 = new Node(49);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      controlFlowAnalysis0.process(node0, node0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, controlFlowAnalysis0, symbolTable0);
      boolean boolean0 = controlFlowAnalysis0.shouldTraverse(nodeTraversal0, node0, node0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0, "com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$2");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, optimizeArgumentsArray0, typedScopeCreator0);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      Node node0 = Node.newString((-4157), "cF^pc+/$\"<O");
      controlFlowAnalysis0.process(node0, node0);
      node0.setType(111);
      boolean boolean0 = controlFlowAnalysis0.shouldTraverse(nodeTraversal0, node0, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = Node.newString("com.google.common.base.CharMatcher$6");
      Node node1 = new Node(115, node0);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      ArrayList<JSType> arrayList0 = new ArrayList<JSType>();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) arrayList0);
      Node node1 = new Node(116, node0);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ArrayList<JSType> arrayList0 = new ArrayList<JSType>();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) arrayList0);
      Node node1 = new Node(117, node0, node0, node0);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis((AbstractCompiler) null, false);
      Node node0 = new Node(118);
      controlFlowAnalysis0.process(node0, node0);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, controlFlowAnalysis0);
      boolean boolean0 = controlFlowAnalysis0.shouldTraverse(nodeTraversal0, node0, node0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = Node.newString(119, "|DMly\"blS", 119, 119);
      node0.addChildToFront(node0);
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
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(105, 2624, 2624);
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
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, controlFlowAnalysis0, syntacticScopeCreator0);
      Node node0 = Node.newString(116, "Y{B:6)f:}K");
      Node node1 = new Node(110, node0, node0, 35, 0);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.visit(nodeTraversal0, node1, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ControlFlowAnalysis", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      Vector<JSType> vector0 = new Vector<JSType>();
      Node node0 = jSTypeRegistry0.createParameters((List<JSType>) vector0);
      node0.setType(111);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Node node0 = Node.newString(115, "msg.no.colon.case");
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
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
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("Not reachable", "Not reachable");
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      controlFlowAnalysis0.process(node0, node0);
      assertEquals(2, Node.SPECIALCALL_WITH);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("Not reachable", "Not reachable");
      node0.addChildAfter(node0, node0);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      controlFlowAnalysis0.process(node0, node0);
      assertEquals(44, Node.IS_OPTIONAL_PARAM);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      ArrayList<JSType> arrayList0 = new ArrayList<JSType>();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) arrayList0);
      SyntheticAst syntheticAst0 = new SyntheticAst("");
      Node node1 = syntheticAst0.getAstRoot(compiler0);
      Node node2 = new Node(77, node0, node0, node1);
      controlFlowAnalysis0.process(node2, node1);
      assertEquals(9, Node.FIXUPS_PROP);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      ScriptOrFnNode scriptOrFnNode0 = (ScriptOrFnNode)compiler0.parseSyntheticCode("com.google.javascript.jscomp.ControlFlowAnalysis$1", "LWJry:VpI2MO");
      Node node0 = new Node(128, scriptOrFnNode0, scriptOrFnNode0, scriptOrFnNode0, scriptOrFnNode0, 120, (-82));
      controlFlowAnalysis0.process(scriptOrFnNode0, node0);
      assertEquals(0, scriptOrFnNode0.getParamCount());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis((AbstractCompiler) null, true);
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator((AbstractCompiler) null);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, controlFlowAnalysis0, syntacticScopeCreator0);
      Node node0 = Node.newString(116, "Y{B:6)f:}K");
      Node node1 = new Node(128, node0, node0, 24, 47);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.visit(nodeTraversal0, node0, node1);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Cannot find break target.
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      Node node0 = new Node(117, 117, 117);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node0);
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
  public void test26()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = Node.newString(117, "com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$2");
      Node node1 = new Node(113, node0, node0, (-310), 126);
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
  public void test27()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = new Node(1235);
      Node node1 = Node.newString(4, " instances of ", 1235, 49);
      controlFlowAnalysis0.process(node0, node1);
      assertEquals(21, Node.LOCALCOUNT_PROP);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.ControlFlowAnalysis$1", "LWJry:VpI2MO");
      Node node1 = new Node(45, node0, node0, node0, node0, 23, 48);
      node0.addChildAfter(node1, node1);
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
  public void test29()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      Node node0 = new Node(114);
      // Undeclared exception!
      try { 
        controlFlowAnalysis0.process(node0, node0);
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
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(30);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      controlFlowAnalysis0.process(node0, node0);
      assertEquals(26, Node.DIRECTCALL_PROP);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$2", "com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$2");
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      controlFlowAnalysis0.process(node0, node0);
      assertEquals(1, Node.SPECIALCALL_EVAL);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      Node node0 = Node.newString("TightenTypes pass appears to be stuck in an infinite loop.");
      Node node1 = new Node(35, node0);
      controlFlowAnalysis0.process(node0, node1);
      assertEquals(29, Node.JSDOC_INFO_PROP);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(37);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      controlFlowAnalysis0.process(node0, node0);
      assertEquals(44, Node.IS_OPTIONAL_PARAM);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = Node.newString(86, "com.google.javascript.jscomp.ControlFlowAnalysis$AstControlFlowGraph$2");
      controlFlowAnalysis0.process(node0, node0);
      assertEquals(42, Node.NO_SIDE_EFFECTS_CALL);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.ControlFlowAnalysis$1", "LWJry:VpI2MO");
      Node node1 = new Node(102);
      controlFlowAnalysis0.process(node0, node1);
      assertEquals(35, Node.QUOTED_PROP);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, false);
      Node node0 = new Node(103, 10, 114);
      controlFlowAnalysis0.process(node0, node0);
      assertFalse(node0.isVarArgs());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.ControlFlowAnalysis$1", "com.google.javascript.jscomp.ControlFlowAnalysis$1");
      Node node1 = new Node(77, node0, node0, node0, node0, (-118), 84);
      boolean boolean0 = ControlFlowAnalysis.isBreakStructure(node1, false);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Node node0 = Node.newString(108, "Od?D#");
      boolean boolean0 = ControlFlowAnalysis.isBreakStructure(node0, false);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Node node0 = Node.newString(115, "msg.no.colon.case");
      boolean boolean0 = ControlFlowAnalysis.isBreakStructure(node0, true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("Not reachable", "Not reachable");
      boolean boolean0 = ControlFlowAnalysis.isBreakStructure(node0, true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Node node0 = new Node(114);
      boolean boolean0 = ControlFlowAnalysis.isContinueStructure(node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Node node0 = Node.newString(115, "msg.no.colon.case");
      boolean boolean0 = ControlFlowAnalysis.isContinueStructure(node0);
      assertTrue(boolean0);
  }
}