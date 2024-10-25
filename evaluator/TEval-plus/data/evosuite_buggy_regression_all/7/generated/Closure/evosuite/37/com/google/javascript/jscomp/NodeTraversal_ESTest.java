/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 17:58:17 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.CheckAccidentalSemicolon;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.CheckSideEffects;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ConvertToDottedProperties;
import com.google.javascript.jscomp.DeadAssignmentsElimination;
import com.google.javascript.jscomp.Denormalize;
import com.google.javascript.jscomp.ExpandJqueryAliases;
import com.google.javascript.jscomp.FieldCleanupPass;
import com.google.javascript.jscomp.FlowSensitiveInlineVariables;
import com.google.javascript.jscomp.JSError;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.jscomp.UnreachableCodeElimination;
import com.google.javascript.jscomp.VarCheck;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NodeTraversal_ESTest extends NodeTraversal_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DeadAssignmentsElimination deadAssignmentsElimination0 = new DeadAssignmentsElimination(compiler0);
      // Undeclared exception!
      try { 
        NodeTraversal.traverseRoots((AbstractCompiler) compiler0, (NodeTraversal.Callback) deadAssignmentsElimination0, (Node[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("8,v");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      VarCheck varCheck0 = new VarCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0, typedScopeCreator0);
      // Undeclared exception!
      try { 
        nodeTraversal0.traverse(node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode(";g8");
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, ";g8");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, fieldCleanupPass_QualifiedNameSearchTraversal0);
      String[] stringArray0 = new String[1];
      JSError jSError0 = nodeTraversal0.makeError(node0, compiler0.OPTIMIZE_LOOP_ERROR, stringArray0);
      assertEquals(1, jSError0.lineNumber);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      FlowSensitiveInlineVariables flowSensitiveInlineVariables0 = new FlowSensitiveInlineVariables(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, flowSensitiveInlineVariables0);
      Node node0 = nodeTraversal0.getCurrentNode();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.NodeTraversal$AbstractNodeTypePruningCallback");
      VarCheck varCheck0 = new VarCheck(compiler0, false);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0);
      // Undeclared exception!
      try { 
        nodeTraversal0.traverse(node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("JSC_STRIP_TYPE_INHERIT_ERROR", "com.google.j<vscripp.jscomp.NodeTraversal$AbstractScopeICallback");
      Scope scope0 = new Scope(node0, compiler0);
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, false);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkSideEffects0);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      assertEquals(1, compiler0.getWarningCount());
      assertEquals("JSC_STRIP_TYPE_INHERIT_ERROR", nodeTraversal0.getSourceName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode(": NULL");
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, ": NULL");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, fieldCleanupPass_QualifiedNameSearchTraversal0);
      nodeTraversal0.traverse(node0);
      assertEquals(" [testcode] ", nodeTraversal0.getSourceName());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("lL~", "lL~");
      Scope scope0 = new Scope(node0, compiler0);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, "lL~");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, fieldCleanupPass_QualifiedNameSearchTraversal0);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      assertEquals("lL~", nodeTraversal0.getSourceName());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("lL~", "lL~");
      Scope scope0 = new Scope(node0, compiler0);
      node0.setType(105);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, "lL~");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, fieldCleanupPass_QualifiedNameSearchTraversal0);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      assertEquals("", nodeTraversal0.getSourceName());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Denormalize.StripConstantAnnotations denormalize_StripConstantAnnotations0 = new Denormalize.StripConstantAnnotations(compiler0);
      Node node0 = Node.newNumber((double) 105, 105, 105);
      node0.setType(105);
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) denormalize_StripConstantAnnotations0);
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
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("8,v6");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      UnreachableCodeElimination unreachableCodeElimination0 = new UnreachableCodeElimination(compiler0, false);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, unreachableCodeElimination0, typedScopeCreator0);
      // Undeclared exception!
      try { 
        nodeTraversal0.traverse(node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DeadAssignmentsElimination deadAssignmentsElimination0 = new DeadAssignmentsElimination(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, deadAssignmentsElimination0);
      Stack<Node> stack0 = new Stack<Node>();
      nodeTraversal0.traverseRoots((List<Node>) stack0);
      assertEquals(0, stack0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("_{BoR%4|;h3h", "_{BoR%4|;h3h");
      VarCheck varCheck0 = new VarCheck(compiler0, true);
      // Undeclared exception!
      try { 
        varCheck0.process(node0, node0);
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
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(0, 0, 0);
      Node node1 = new Node(15, node0);
      Normalize.PropagateConstantAnnotationsOverVars normalize_PropagateConstantAnnotationsOverVars0 = new Normalize.PropagateConstantAnnotationsOverVars(compiler0, true);
      // Undeclared exception!
      try { 
        normalize_PropagateConstantAnnotationsOverVars0.process(node0, node1);
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
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      FlowSensitiveInlineVariables flowSensitiveInlineVariables0 = new FlowSensitiveInlineVariables(compiler0);
      Node node0 = compiler0.parseSyntheticCode("lL~", "lL~");
      Scope scope0 = new Scope(node0, compiler0);
      node0.setType(105);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, flowSensitiveInlineVariables0);
      // Undeclared exception!
      try { 
        nodeTraversal0.traverseAtScope(scope0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckAccidentalSemicolon checkAccidentalSemicolon0 = new CheckAccidentalSemicolon(checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkAccidentalSemicolon0);
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.NodeTraversal$AbsEractNodeTypePruningCalQback", "com.google.javascript.jscomp.NodeTraversal$AbsEractNodeTypePruningCalQback");
      nodeTraversal0.traverse(node0);
      int int0 = nodeTraversal0.getLineNumber();
      assertEquals("com.google.javascript.jscomp.NodeTraversal$AbsEractNodeTypePruningCalQback", nodeTraversal0.getSourceName());
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      FlowSensitiveInlineVariables flowSensitiveInlineVariables0 = new FlowSensitiveInlineVariables(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, flowSensitiveInlineVariables0);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ArrayList<JSType> arrayList0 = new ArrayList<JSType>();
      Node node0 = jSTypeRegistry0.createParameters((List<JSType>) arrayList0);
      nodeTraversal0.traverse(node0);
      int int0 = nodeTraversal0.getLineNumber();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DeadAssignmentsElimination deadAssignmentsElimination0 = new DeadAssignmentsElimination(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, deadAssignmentsElimination0);
      Node node0 = compiler0.parseTestCode("S0Ep\"`bJu:bbQ\"[G");
      nodeTraversal0.traverseInnerNode(node0, node0, (Scope) null);
      nodeTraversal0.getModule();
      assertEquals(" [testcode] ", nodeTraversal0.getSourceName());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ExpandJqueryAliases expandJqueryAliases0 = new ExpandJqueryAliases(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, expandJqueryAliases0);
      compiler0.parseTestCode("");
      JSModule jSModule0 = nodeTraversal0.getModule();
      assertNull(jSModule0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      FlowSensitiveInlineVariables flowSensitiveInlineVariables0 = new FlowSensitiveInlineVariables(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, flowSensitiveInlineVariables0);
      Node node0 = nodeTraversal0.getEnclosingFunction();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DeadAssignmentsElimination deadAssignmentsElimination0 = new DeadAssignmentsElimination(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, deadAssignmentsElimination0);
      // Undeclared exception!
      try { 
        nodeTraversal0.traverse((Node) null);
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
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DeadAssignmentsElimination deadAssignmentsElimination0 = new DeadAssignmentsElimination(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, deadAssignmentsElimination0);
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.CleanupPasses$3", "com.google.javascript.jscomp.CleanupPasses$3");
      Scope scope0 = new Scope(node0, compiler0);
      // Undeclared exception!
      try { 
        nodeTraversal0.traverseInnerNode((Node) null, node0, scope0);
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
      Compiler compiler0 = new Compiler();
      DeadAssignmentsElimination deadAssignmentsElimination0 = new DeadAssignmentsElimination(compiler0);
      Node node0 = compiler0.parseSyntheticCode("wnk4:tJMLu&0Xcti$A#", "wnk4:tJMLu&0Xcti$A#");
      Scope scope0 = new Scope(node0, compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, deadAssignmentsElimination0);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      assertEquals("wnk4:tJMLu&0Xcti$A#", nodeTraversal0.getSourceName());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DeadAssignmentsElimination deadAssignmentsElimination0 = new DeadAssignmentsElimination(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, deadAssignmentsElimination0);
      // Undeclared exception!
      try { 
        nodeTraversal0.getControlFlowGraph();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        Normalize.parseAndNormalizeTestCode(compiler0, "D!&0dAcd|5Ot@xTr", "D!&0dAcd|5Ot@xTr");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DeadAssignmentsElimination deadAssignmentsElimination0 = new DeadAssignmentsElimination(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, deadAssignmentsElimination0);
      boolean boolean0 = nodeTraversal0.hasScope();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode(": NULL");
      Scope scope0 = new Scope(node0, compiler0);
      Node node1 = new Node(105, node0, node0, node0, 42, 53);
      node0.srcrefTree(node1);
      ConvertToDottedProperties convertToDottedProperties0 = new ConvertToDottedProperties(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, convertToDottedProperties0);
      nodeTraversal0.traverseAtScope(scope0);
      assertEquals(42, nodeTraversal0.getLineNumber());
  }
}
