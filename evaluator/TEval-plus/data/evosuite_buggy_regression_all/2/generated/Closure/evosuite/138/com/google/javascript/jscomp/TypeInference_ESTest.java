/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:38:42 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.collect.Multimap;
import com.google.javascript.jscomp.ClosureCodingConvention;
import com.google.javascript.jscomp.ClosureReverseAbstractInterpreter;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ControlFlowGraph;
import com.google.javascript.jscomp.FlowScope;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TypeInference;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.BooleanLiteralSet;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.List;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeInference_ESTest extends TypeInference_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>((Node) null);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, jSTypeRegistry0);
      Scope scope0 = new Scope((Node) null, compiler0);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0);
      FlowScope flowScope0 = typeInference0.createEntryLattice();
      // Undeclared exception!
      try { 
        typeInference0.branchedFlowThrough((Node) null, flowScope0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.TypeInference", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>((Node) null);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, jSTypeRegistry0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope((Node) null);
      Stack<Scope.Var> stack0 = new Stack<Scope.Var>();
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, stack0);
      FlowScope flowScope0 = typeInference0.createInitialEstimateLattice();
      List<FlowScope> list0 = typeInference0.branchedFlowThrough((Node) null, flowScope0);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>((Node) null);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, jSTypeRegistry0);
      Scope scope0 = new Scope((Node) null, compiler0);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0);
      Multimap<Scope, Scope.Var> multimap0 = typeInference0.getAssignedOuterLocalVars();
      assertNotNull(multimap0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>((Node) null);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(closureCodingConvention0, jSTypeRegistry0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope((Node) null);
      Stack<Scope.Var> stack0 = new Stack<Scope.Var>();
      Scope.Var scope_Var0 = scope0.getVar("String.prototype");
      stack0.add(scope_Var0);
      TypeInference typeInference0 = new TypeInference(compiler0, controlFlowGraph0, closureReverseAbstractInterpreter0, scope0, stack0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      BooleanLiteralSet booleanLiteralSet0 = BooleanLiteralSet.TRUE;
      BooleanLiteralSet booleanLiteralSet1 = TypeInference.getBooleanOutcomes(booleanLiteralSet0, booleanLiteralSet0, true);
      assertSame(booleanLiteralSet1, booleanLiteralSet0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      BooleanLiteralSet booleanLiteralSet0 = BooleanLiteralSet.TRUE;
      BooleanLiteralSet booleanLiteralSet1 = TypeInference.getBooleanOutcomes(booleanLiteralSet0, booleanLiteralSet0, false);
      assertEquals(BooleanLiteralSet.TRUE, booleanLiteralSet1);
  }
}