/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:17:07 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.collect.Multimap;
import com.google.javascript.jscomp.ClosureReverseAbstractInterpreter;
import com.google.javascript.jscomp.CodingConvention;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ControlFlowGraph;
import com.google.javascript.jscomp.DefaultCodingConvention;
import com.google.javascript.jscomp.FlowScope;
import com.google.javascript.jscomp.GoogleCodingConvention;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.LinkedFlowScope;
import com.google.javascript.jscomp.PrintStreamErrorManager;
import com.google.javascript.jscomp.ReverseAbstractInterpreter;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.SemanticReverseAbstractInterpreter;
import com.google.javascript.jscomp.TightenTypes;
import com.google.javascript.jscomp.TypeInference;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.BooleanLiteralSet;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.ArrayDeque;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeInference_ESTest extends TypeInference_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(17, "Y1H]:hMg", 1142, 17);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0);
      DefaultCodingConvention defaultCodingConvention0 = new DefaultCodingConvention();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      Scope scope0 = new Scope(node0, compiler0);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(defaultCodingConvention0, jSTypeRegistry0);
      ArrayDeque<Scope.Var> arrayDeque0 = new ArrayDeque<Scope.Var>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, semanticReverseAbstractInterpreter0, scope0, arrayDeque0);
      FlowScope flowScope0 = typeInference0.createEntryLattice();
      List<FlowScope> list0 = typeInference0.branchedFlowThrough(node0, flowScope0);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("62ok");
      PrintStreamErrorManager printStreamErrorManager0 = new PrintStreamErrorManager(mockPrintStream0);
      Compiler compiler0 = new Compiler(printStreamErrorManager0);
      Node node0 = compiler0.parseTestCode("62ok");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0);
      Scope scope0 = new Scope(node0, compiler0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      ReverseAbstractInterpreter reverseAbstractInterpreter0 = compiler0.getReverseAbstractInterpreter();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, reverseAbstractInterpreter0, scope0);
      Node node1 = new Node(16, node0, node0, (-688), 45);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node1, linkedFlowScope0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // GT : com.google.javascript.rhino.jstype.BooleanType@0000000042 does not exist in graph
         //
         verifyException("com.google.javascript.jscomp.graph.LinkedDirectedGraph", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ZyZafp%%v6T");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0);
      Scope scope0 = new Scope(node0, compiler0);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, (ReverseAbstractInterpreter) null, scope0);
      Multimap<Scope, Scope.Var> multimap0 = typeInference0.getAssignedOuterLocalVars();
      assertNotNull(multimap0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("62ok");
      PrintStreamErrorManager printStreamErrorManager0 = new PrintStreamErrorManager(mockPrintStream0);
      Compiler compiler0 = new Compiler(printStreamErrorManager0);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("62ok");
      Node node0 = compiler0.parse(jSSourceFile0);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0);
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, jSTypeRegistry0);
      Scope scope0 = new Scope(node0, compiler0);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, semanticReverseAbstractInterpreter0, scope0);
      FlowScope flowScope0 = typeInference0.createInitialEstimateLattice();
      List<FlowScope> list0 = typeInference0.branchedFlowThrough(node0, flowScope0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("d]F6in");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0);
      Scope scope0 = new Scope(node0, compiler0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      ReverseAbstractInterpreter reverseAbstractInterpreter0 = compiler0.getReverseAbstractInterpreter();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, reverseAbstractInterpreter0, scope0);
      Node node1 = new Node(5, node0, node0, 29, 30);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node1, linkedFlowScope0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // GOTO 29 does not exist in graph
         //
         verifyException("com.google.javascript.jscomp.graph.LinkedDirectedGraph", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("62ok");
      PrintStreamErrorManager printStreamErrorManager0 = new PrintStreamErrorManager(mockPrintStream0);
      Compiler compiler0 = new Compiler(printStreamErrorManager0);
      Node node0 = compiler0.parseTestCode("62ok");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0);
      Scope scope0 = new Scope(node0, compiler0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, (ReverseAbstractInterpreter) null, scope0);
      Node node1 = new Node(6, node0, node0, 16, (-4106));
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node1, linkedFlowScope0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // IFEQ does not exist in graph
         //
         verifyException("com.google.javascript.jscomp.graph.LinkedDirectedGraph", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("62ok");
      PrintStreamErrorManager printStreamErrorManager0 = new PrintStreamErrorManager(mockPrintStream0);
      Compiler compiler0 = new Compiler(printStreamErrorManager0);
      Node node0 = compiler0.parseTestCode("62ok");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0);
      Scope scope0 = new Scope(node0, compiler0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, (ReverseAbstractInterpreter) null, scope0);
      Node node1 = new Node(24, node0, node0, 35, (-1040));
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node1, linkedFlowScope0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DIV : com.google.javascript.rhino.jstype.NumberType@0000000044 does not exist in graph
         //
         verifyException("com.google.javascript.jscomp.graph.LinkedDirectedGraph", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("62ok");
      PrintStreamErrorManager printStreamErrorManager0 = new PrintStreamErrorManager(mockPrintStream0);
      Compiler compiler0 = new Compiler(printStreamErrorManager0);
      Node node0 = compiler0.parseTestCode("JSC_TEMPLATE_TYPE_OF_THIS_EXPECTED");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0);
      Scope scope0 = new Scope(node0, compiler0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      ReverseAbstractInterpreter reverseAbstractInterpreter0 = compiler0.getReverseAbstractInterpreter();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, reverseAbstractInterpreter0, scope0);
      Node node1 = new Node(25, node0, node0, (-1501), 90);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node1, linkedFlowScope0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // MOD : com.google.javascript.rhino.jstype.NumberType@0000000044 does not exist in graph
         //
         verifyException("com.google.javascript.jscomp.graph.LinkedDirectedGraph", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("62ok");
      PrintStreamErrorManager printStreamErrorManager0 = new PrintStreamErrorManager(mockPrintStream0);
      Compiler compiler0 = new Compiler(printStreamErrorManager0);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>((Node) null);
      DefaultCodingConvention defaultCodingConvention0 = new DefaultCodingConvention();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(defaultCodingConvention0, jSTypeRegistry0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, defaultCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope((Node) null);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      Stack<JSType> stack0 = new Stack<JSType>();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) stack0);
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node0, linkedFlowScope0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.TypeInference", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("62ok");
      PrintStreamErrorManager printStreamErrorManager0 = new PrintStreamErrorManager(mockPrintStream0);
      Compiler compiler0 = new Compiler(printStreamErrorManager0);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("62ok");
      Node node0 = compiler0.parse(jSSourceFile0);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0);
      CodingConvention codingConvention0 = compiler0.getCodingConvention();
      LinkedHashSet<Scope.Var> linkedHashSet0 = new LinkedHashSet<Scope.Var>();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(codingConvention0, jSTypeRegistry0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, semanticReverseAbstractInterpreter0, scope0, linkedHashSet0);
      TypeInference typeInference1 = new TypeInference(compiler0, controlFlowGraph0, semanticReverseAbstractInterpreter0, scope0);
      Node node1 = new Node(98, node0, node0, 127, (-3508));
      FlowScope flowScope0 = typeInference1.createInitialEstimateLattice();
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough(node1, flowScope0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.TypeInference", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("62ok");
      PrintStreamErrorManager printStreamErrorManager0 = new PrintStreamErrorManager(mockPrintStream0);
      Compiler compiler0 = new Compiler(printStreamErrorManager0);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("62ok");
      Node node0 = compiler0.parse(jSSourceFile0);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0);
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, jSTypeRegistry0);
      Scope scope0 = new Scope(node0, compiler0);
      ArrayDeque<Scope.Var> arrayDeque0 = new ArrayDeque<Scope.Var>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, semanticReverseAbstractInterpreter0, scope0, arrayDeque0);
      Node node1 = new Node(122, node0, node0, 0, (-1441));
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      FlowScope flowScope0 = typeInference0.flowThrough(node1, linkedFlowScope0);
      assertNotSame(flowScope0, linkedFlowScope0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BooleanLiteralSet booleanLiteralSet0 = BooleanLiteralSet.EMPTY;
      BooleanLiteralSet booleanLiteralSet1 = TypeInference.getBooleanOutcomes(booleanLiteralSet0, booleanLiteralSet0, true);
      assertSame(booleanLiteralSet0, booleanLiteralSet1);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BooleanLiteralSet booleanLiteralSet0 = BooleanLiteralSet.EMPTY;
      BooleanLiteralSet booleanLiteralSet1 = TypeInference.getBooleanOutcomes(booleanLiteralSet0, booleanLiteralSet0, false);
      assertEquals(BooleanLiteralSet.EMPTY, booleanLiteralSet1);
  }
}