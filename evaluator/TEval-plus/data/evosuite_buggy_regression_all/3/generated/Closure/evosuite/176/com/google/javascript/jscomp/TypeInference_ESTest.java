/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:25:13 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.collect.ImmutableSortedMap;
import com.google.javascript.jscomp.ClosureCodingConvention;
import com.google.javascript.jscomp.CodingConvention;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ControlFlowGraph;
import com.google.javascript.jscomp.LinkedFlowScope;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.PeepholeSubstituteAlternateSyntax;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TightenTypes;
import com.google.javascript.jscomp.TypeInference;
import com.google.javascript.jscomp.type.ClosureReverseAbstractInterpreter;
import com.google.javascript.jscomp.type.FlowScope;
import com.google.javascript.jscomp.type.ReverseAbstractInterpreter;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.BooleanLiteralSet;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.HashMap;
import java.util.List;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeInference_ESTest extends TypeInference_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(12, node0, node0, node0, (-1587), 2);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, false);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, jSTypeRegistry0);
      HashMap<String, CodingConvention.AssertionFunctionSpec> hashMap0 = new HashMap<String, CodingConvention.AssertionFunctionSpec>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, hashMap0);
      typeInference0.analyze(57);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ImmutableSortedMap<String, CodingConvention.AssertionFunctionSpec> immutableSortedMap0 = ImmutableSortedMap.of();
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(674, node0, node0, node0, 48, 0);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      ReverseAbstractInterpreter reverseAbstractInterpreter0 = compiler0.getReverseAbstractInterpreter();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, reverseAbstractInterpreter0, scope0, immutableSortedMap0);
      FlowScope flowScope0 = typeInference0.createInitialEstimateLattice();
      FlowScope flowScope1 = typeInference0.flowThrough(node0, flowScope0);
      assertSame(flowScope1, flowScope0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "lu1f]?Ihk2|yO");
      Node node1 = new Node(4, node0, node0);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>((Node) null, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, jSTypeRegistry0);
      HashMap<String, CodingConvention.AssertionFunctionSpec> hashMap0 = new HashMap<String, CodingConvention.AssertionFunctionSpec>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, hashMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      FlowScope flowScope0 = typeInference0.flowThrough(node1, linkedFlowScope0);
      assertNotSame(flowScope0, linkedFlowScope0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "lu1f]?Ihk2|yO");
      Node node1 = new Node(5, node0, node0, node0, 5, 8);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>((Node) null, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, jSTypeRegistry0);
      HashMap<String, CodingConvention.AssertionFunctionSpec> hashMap0 = new HashMap<String, CodingConvention.AssertionFunctionSpec>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, hashMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      FlowScope flowScope0 = typeInference0.flowThrough(node1, linkedFlowScope0);
      assertNotSame(flowScope0, linkedFlowScope0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ImmutableSortedMap<String, CodingConvention.AssertionFunctionSpec> immutableSortedMap0 = ImmutableSortedMap.of();
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "lu1f]?Ihk2|yO");
      Node node1 = new Node(5, node0, node0, node0, 5, 8);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, (JSTypeRegistry) null);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, immutableSortedMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      Node node2 = Node.newString(8, "", 15, 2787);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node2, linkedFlowScope0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 8
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(37, node0, node0, node0, node0, 38, 50);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, (JSTypeRegistry) null);
      HashMap<String, CodingConvention.AssertionFunctionSpec> hashMap0 = new HashMap<String, CodingConvention.AssertionFunctionSpec>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, hashMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      Node node2 = new Node(12, node1, 2, 54);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node2, linkedFlowScope0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // EQ 2 : boolean does not exist in graph
         //
         verifyException("com.google.javascript.jscomp.graph.Graph", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(15, node0, node0, node0);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, (JSTypeRegistry) null);
      HashMap<String, CodingConvention.AssertionFunctionSpec> hashMap0 = new HashMap<String, CodingConvention.AssertionFunctionSpec>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, hashMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      Node node2 = new Node(12, node1, 2, 54);
      FlowScope flowScope0 = typeInference0.flowThrough(node2, linkedFlowScope0);
      assertNotSame(flowScope0, linkedFlowScope0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ImmutableSortedMap<String, CodingConvention.AssertionFunctionSpec> immutableSortedMap0 = ImmutableSortedMap.of();
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(674, node0, node0, node0, 48, 54);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      Node node2 = new Node(30, node1, 0, 39);
      ReverseAbstractInterpreter reverseAbstractInterpreter0 = compiler0.getReverseAbstractInterpreter();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, reverseAbstractInterpreter0, scope0, immutableSortedMap0);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node2, linkedFlowScope0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // NEW 0 does not exist in graph
         //
         verifyException("com.google.javascript.jscomp.graph.Graph", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(674, node0, node0, node0, 48, 0);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, (JSTypeRegistry) null);
      HashMap<String, CodingConvention.AssertionFunctionSpec> hashMap0 = new HashMap<String, CodingConvention.AssertionFunctionSpec>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, hashMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      Node node2 = new Node(31, node1, 0, 36);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node2, linkedFlowScope0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DELPROP 0 : boolean does not exist in graph
         //
         verifyException("com.google.javascript.jscomp.graph.Graph", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ImmutableSortedMap<String, CodingConvention.AssertionFunctionSpec> immutableSortedMap0 = ImmutableSortedMap.of();
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(46, node0, node0, node0, (-18), 54);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, (JSTypeRegistry) null);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, immutableSortedMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      Node node2 = new Node(36, node1, 57, 53);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node2, linkedFlowScope0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 36
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ImmutableSortedMap<String, CodingConvention.AssertionFunctionSpec> immutableSortedMap0 = ImmutableSortedMap.of();
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "cC`?+)");
      Node node1 = new Node(166, node0, node0, node0, 42, 15);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, false);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      Node node2 = Node.newString("cC`?+)");
      ReverseAbstractInterpreter reverseAbstractInterpreter0 = compiler0.getReverseAbstractInterpreter();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, reverseAbstractInterpreter0, scope0, immutableSortedMap0);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node2, linkedFlowScope0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // STRING cC`?+) does not exist in graph
         //
         verifyException("com.google.javascript.jscomp.graph.Graph", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ImmutableSortedMap<String, CodingConvention.AssertionFunctionSpec> immutableSortedMap0 = ImmutableSortedMap.of();
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(42, node0, node0, node0, 2, 29);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node1, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, (JSTypeRegistry) null);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, immutableSortedMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      List<FlowScope> list0 = typeInference0.branchedFlowThrough(node1, linkedFlowScope0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ImmutableSortedMap<String, CodingConvention.AssertionFunctionSpec> immutableSortedMap0 = ImmutableSortedMap.of();
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(46, node0, node0, node0, 46, 54);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, (JSTypeRegistry) null);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, immutableSortedMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node1, linkedFlowScope0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // SHNE 46 : boolean does not exist in graph
         //
         verifyException("com.google.javascript.jscomp.graph.Graph", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(49, node0, node0, node0, 118, 54);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node1, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, (JSTypeRegistry) null);
      HashMap<String, CodingConvention.AssertionFunctionSpec> hashMap0 = new HashMap<String, CodingConvention.AssertionFunctionSpec>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, hashMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      Node node2 = new Node(51, node1, 1076, 56);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node2, linkedFlowScope0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // IN 1076 : boolean does not exist in graph
         //
         verifyException("com.google.javascript.jscomp.graph.Graph", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(49, node0, node0, node0, 118, 54);
      Scope scope0 = Scope.createGlobalScope(node1);
      Node node2 = new Node(55, node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node1, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, (JSTypeRegistry) null);
      HashMap<String, CodingConvention.AssertionFunctionSpec> hashMap0 = new HashMap<String, CodingConvention.AssertionFunctionSpec>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, hashMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node2, linkedFlowScope0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 55
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      Stack<JSType> stack0 = new Stack<JSType>();
      Node node1 = jSTypeRegistry0.createParameters((List<JSType>) stack0);
      Node node2 = new Node(30, node0, node1, node1, 3, 57);
      Scope scope0 = Scope.createGlobalScope(node2);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, (JSTypeRegistry) null);
      HashMap<String, CodingConvention.AssertionFunctionSpec> hashMap0 = new HashMap<String, CodingConvention.AssertionFunctionSpec>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, hashMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      // Undeclared exception!
      try { 
        typeInference0.flowThrough(node1, linkedFlowScope0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.TypeInference", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ImmutableSortedMap<String, CodingConvention.AssertionFunctionSpec> immutableSortedMap0 = ImmutableSortedMap.of();
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "lu1f]?Ihk2|yO");
      Node node1 = new Node(5, node0, node0, node0, 5, 8);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node1, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, jSTypeRegistry0);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, immutableSortedMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice((Scope) null);
      Node node2 = new Node(108, node1, (-1917), (-8));
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node2, linkedFlowScope0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // IF does not exist in graph
         //
         verifyException("com.google.javascript.jscomp.graph.Graph", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ImmutableSortedMap<String, CodingConvention.AssertionFunctionSpec> immutableSortedMap0 = ImmutableSortedMap.of();
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(111, node0, node0, node0, 2, 118);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node1, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, (JSTypeRegistry) null);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, immutableSortedMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      List<FlowScope> list0 = typeInference0.branchedFlowThrough(node1, linkedFlowScope0);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(674, node0, node0, node0, 48, 0);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, (JSTypeRegistry) null);
      HashMap<String, CodingConvention.AssertionFunctionSpec> hashMap0 = new HashMap<String, CodingConvention.AssertionFunctionSpec>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, hashMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      node1.setType(113);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node1, linkedFlowScope0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // WHILE 48 does not exist in graph
         //
         verifyException("com.google.javascript.jscomp.graph.Graph", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ImmutableSortedMap<String, CodingConvention.AssertionFunctionSpec> immutableSortedMap0 = ImmutableSortedMap.of();
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(49, node0, node0, node0, 118, 54);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node1, true, true);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      Node node2 = new Node(119, node1, 122, 32);
      ReverseAbstractInterpreter reverseAbstractInterpreter0 = compiler0.getReverseAbstractInterpreter();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, reverseAbstractInterpreter0, scope0, immutableSortedMap0);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node2, linkedFlowScope0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // WITH 122 does not exist in graph
         //
         verifyException("com.google.javascript.jscomp.graph.Graph", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ImmutableSortedMap<String, CodingConvention.AssertionFunctionSpec> immutableSortedMap0 = ImmutableSortedMap.of();
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "lu1f]?Ihk2|yO");
      Node node1 = new Node(5, node0, node0, node0, 5, 8);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node1, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, jSTypeRegistry0);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, immutableSortedMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      Node node2 = new Node(120, node1, 53, 15);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node2, linkedFlowScope0);
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
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(674, node0, node0, node0, 48, 0);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, (JSTypeRegistry) null);
      HashMap<String, CodingConvention.AssertionFunctionSpec> hashMap0 = new HashMap<String, CodingConvention.AssertionFunctionSpec>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, hashMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      node1.setType(128);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node1, linkedFlowScope0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 128
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(12, node0, node0, node0, (-1587), 2);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, false);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, jSTypeRegistry0);
      HashMap<String, CodingConvention.AssertionFunctionSpec> hashMap0 = new HashMap<String, CodingConvention.AssertionFunctionSpec>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, hashMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      Node node2 = Node.newString(134, "goog.asserts.assertElement", 33, (-2128));
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node2, linkedFlowScope0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 134
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ImmutableSortedMap<String, CodingConvention.AssertionFunctionSpec> immutableSortedMap0 = ImmutableSortedMap.of();
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(115, node0, node0, node0, node0, (-20), 1982);
      PeepholeSubstituteAlternateSyntax peepholeSubstituteAlternateSyntax0 = new PeepholeSubstituteAlternateSyntax(true);
      Node node2 = peepholeSubstituteAlternateSyntax0.optimizeSubtree(node1);
      Scope scope0 = Scope.createGlobalScope(node2);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, (JSTypeRegistry) null);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, immutableSortedMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      Node node3 = new Node(135, node2, 121, 163);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node3, linkedFlowScope0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 135
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ImmutableSortedMap<String, CodingConvention.AssertionFunctionSpec> immutableSortedMap0 = ImmutableSortedMap.of();
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(49, node0, node0, node0, 118, 54);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node1, true, true);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      Node node2 = new Node(136, node1, 122, 32);
      ReverseAbstractInterpreter reverseAbstractInterpreter0 = compiler0.getReverseAbstractInterpreter();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, reverseAbstractInterpreter0, scope0, immutableSortedMap0);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node2, linkedFlowScope0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 136
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "");
      Node node1 = new Node(674, node0, node0, node0, 48, 2);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, (JSTypeRegistry) null);
      HashMap<String, CodingConvention.AssertionFunctionSpec> hashMap0 = new HashMap<String, CodingConvention.AssertionFunctionSpec>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, hashMap0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      Node node2 = new Node(137, node1, 0, 54);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node2, linkedFlowScope0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 137
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ImmutableSortedMap<String, CodingConvention.AssertionFunctionSpec> immutableSortedMap0 = ImmutableSortedMap.of();
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "cC`?+)");
      Node node1 = new Node(143, node0, node0, node0, (-55), 15);
      Scope scope0 = Scope.createGlobalScope(node1);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      ReverseAbstractInterpreter reverseAbstractInterpreter0 = compiler0.getReverseAbstractInterpreter();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, reverseAbstractInterpreter0, scope0, immutableSortedMap0);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node1, linkedFlowScope0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 143
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      BooleanLiteralSet booleanLiteralSet0 = BooleanLiteralSet.FALSE;
      BooleanLiteralSet booleanLiteralSet1 = TypeInference.getBooleanOutcomes(booleanLiteralSet0, booleanLiteralSet0, true);
      assertSame(booleanLiteralSet1, booleanLiteralSet0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      BooleanLiteralSet booleanLiteralSet0 = BooleanLiteralSet.BOTH;
      BooleanLiteralSet booleanLiteralSet1 = TypeInference.getBooleanOutcomes(booleanLiteralSet0, booleanLiteralSet0, false);
      assertSame(booleanLiteralSet0, booleanLiteralSet1);
  }
}