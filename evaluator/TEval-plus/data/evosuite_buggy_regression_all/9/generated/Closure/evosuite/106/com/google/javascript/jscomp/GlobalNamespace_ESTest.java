/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:01:19 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ExternExportsPass;
import com.google.javascript.jscomp.GlobalNamespace;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.SyntheticAst;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeNative;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.ObjectType;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class GlobalNamespace_ESTest extends GlobalNamespace_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("com.google.javascript.jscomp.mozilla.rhino.ast.WithStatement", (GlobalNamespace.Name) null, false);
      String string0 = globalNamespace_Name0.toString();
      assertEquals("com.google.javascript.jscomp.mozilla.rhino.ast.WithStatement (OTHER): globalSets=0, localSets=0, totalGets=0, aliasingGets=0, callGets=0", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ExternExportsPass externExportsPass0 = new ExternExportsPass(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, externExportsPass0);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.SET_FROM_LOCAL;
      GlobalNamespace.Ref globalNamespace_Ref0 = null;
      try {
        globalNamespace_Ref0 = new GlobalNamespace.Ref(nodeTraversal0, (Node) null, globalNamespace_Ref_Type0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.ALIASING_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      GlobalNamespace.Ref globalNamespace_Ref1 = globalNamespace_Ref0.getTwin();
      assertNull(globalNamespace_Ref1);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      JSType[] jSTypeArray0 = new JSType[7];
      Node node0 = jSTypeRegistry0.createParameters(jSTypeArray0);
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, node0, node0);
      globalNamespace0.getNameIndex();
      List<GlobalNamespace.Name> list0 = globalNamespace0.getNameForest();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      JSType[] jSTypeArray0 = new JSType[7];
      Node node0 = jSTypeRegistry0.createParameters(jSTypeArray0);
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, node0, node0);
      globalNamespace0.getNameIndex();
      Map<String, GlobalNamespace.Name> map0 = globalNamespace0.getNameIndex();
      assertTrue(map0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.AnalyzePrototypeProperties$ProcessExternProperties");
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      JSTypeNative jSTypeNative0 = JSTypeNative.NO_OBJECT_TYPE;
      ObjectType objectType0 = jSTypeRegistry0.getNativeObjectType(jSTypeNative0);
      Scope scope0 = new Scope(node0, objectType0);
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, node0);
      LinkedHashSet<Node> linkedHashSet0 = new LinkedHashSet<Node>();
      globalNamespace0.scanNewNodes(scope0, linkedHashSet0);
      assertEquals(0, linkedHashSet0.size());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Node node0 = Node.newNumber(2.147483647E9);
      Compiler compiler0 = new Compiler();
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, node0);
      List<GlobalNamespace.Name> list0 = globalNamespace0.getNameForest();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.AnalyzePrototypeProperties$ProcessExternProperties");
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, node0, node0);
      List<GlobalNamespace.Name> list0 = globalNamespace0.getNameForest();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString("f,F(t");
      GlobalNamespace globalNamespace0 = new GlobalNamespace(compiler0, node0, node0);
      List<GlobalNamespace.Name> list0 = globalNamespace0.getNameForest();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Name globalNamespace_Name1 = globalNamespace_Name0.addProperty("", false);
      GlobalNamespace.Name globalNamespace_Name2 = globalNamespace_Name0.addProperty("", false);
      assertNotNull(globalNamespace_Name2);
      assertNotSame(globalNamespace_Name2, globalNamespace_Name1);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, false);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.SET_FROM_GLOBAL;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.DIRECT_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.PROTOTYPE_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("? d", (GlobalNamespace.Name) null, false);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.CALL_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, false);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.SET_FROM_GLOBAL;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.DIRECT_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.CALL_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.declaration = globalNamespace_Ref0;
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, false);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.SET_FROM_LOCAL;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.ALIASING_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("#2V'n=", (GlobalNamespace.Name) null, false);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
      globalNamespace_Name0.removeRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.DIRECT_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, true);
      boolean boolean0 = globalNamespace_Name0.canEliminate();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, true);
      boolean boolean0 = globalNamespace_Name0.canCollapse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, false);
      boolean boolean0 = globalNamespace_Name0.canCollapse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, false);
      globalNamespace_Name0.setIsClassOrEnum();
      boolean boolean0 = globalNamespace_Name0.canCollapse();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, true);
      GlobalNamespace.Name globalNamespace_Name1 = globalNamespace_Name0.addProperty("", false);
      assertNotNull(globalNamespace_Name1);
      
      boolean boolean0 = globalNamespace_Name1.canCollapse();
      assertNotSame(globalNamespace_Name1, globalNamespace_Name0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, false);
      globalNamespace_Name0.globalSets = 4816;
      boolean boolean0 = globalNamespace_Name0.canCollapse();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, false);
      globalNamespace_Name0.localSets = 1553;
      boolean boolean0 = globalNamespace_Name0.canCollapse();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, true);
      globalNamespace_Name0.globalSets = 13;
      boolean boolean0 = globalNamespace_Name0.needsToBeStubbed();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, true);
      boolean boolean0 = globalNamespace_Name0.needsToBeStubbed();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, true);
      globalNamespace_Name0.localSets = 86;
      boolean boolean0 = globalNamespace_Name0.needsToBeStubbed();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name(")pj,tIh,q<2606", (GlobalNamespace.Name) null, false);
      GlobalNamespace.Name globalNamespace_Name1 = new GlobalNamespace.Name(")pj,tIh,q<2606", globalNamespace_Name0, false);
      globalNamespace_Name1.setIsClassOrEnum();
      boolean boolean0 = globalNamespace_Name0.isNamespace();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, true);
      boolean boolean0 = globalNamespace_Name0.isNamespace();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, false);
      GlobalNamespace.Name globalNamespace_Name1 = new GlobalNamespace.Name("", globalNamespace_Name0, false);
      boolean boolean0 = globalNamespace_Name1.isSimpleName();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("KAKt\"PK<", (GlobalNamespace.Name) null, true);
      boolean boolean0 = globalNamespace_Name0.isSimpleName();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name(")pj,tIh,q<2606", (GlobalNamespace.Name) null, false);
      GlobalNamespace.Name globalNamespace_Name1 = new GlobalNamespace.Name(")pj,tIh,q<2606", globalNamespace_Name0, false);
      String string0 = globalNamespace_Name1.fullName();
      assertEquals(")pj,tIh,q<2606.)pj,tIh,q<2606", string0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      GlobalNamespace.Name globalNamespace_Name0 = new GlobalNamespace.Name("", (GlobalNamespace.Name) null, false);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.SET_FROM_GLOBAL;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      SyntheticAst syntheticAst0 = new SyntheticAst("");
      Compiler compiler0 = new Compiler();
      Node node0 = syntheticAst0.getAstRoot(compiler0);
      Node node1 = new Node(42, node0, node0, node0, 42, 24);
      globalNamespace_Ref0.node = node0;
      globalNamespace_Name0.addRef(globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.SET_FROM_GLOBAL;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      boolean boolean0 = globalNamespace_Ref0.isSet();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.SET_FROM_LOCAL;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      boolean boolean0 = globalNamespace_Ref0.isSet();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.ALIASING_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      boolean boolean0 = globalNamespace_Ref0.isSet();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.ALIASING_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      // Undeclared exception!
      try { 
        GlobalNamespace.Ref.markTwins(globalNamespace_Ref0, globalNamespace_Ref0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.ALIASING_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type1 = GlobalNamespace.Ref.Type.SET_FROM_GLOBAL;
      GlobalNamespace.Ref globalNamespace_Ref1 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type1);
      GlobalNamespace.Ref.markTwins(globalNamespace_Ref1, globalNamespace_Ref0);
      assertNotSame(globalNamespace_Ref1, globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.DIRECT_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      // Undeclared exception!
      try { 
        GlobalNamespace.Ref.markTwins(globalNamespace_Ref0, globalNamespace_Ref0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.ALIASING_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type1 = GlobalNamespace.Ref.Type.SET_FROM_GLOBAL;
      GlobalNamespace.Ref globalNamespace_Ref1 = globalNamespace_Ref0.cloneAndReclassify(globalNamespace_Ref_Type1);
      GlobalNamespace.Ref.markTwins(globalNamespace_Ref0, globalNamespace_Ref1);
      assertNotSame(globalNamespace_Ref1, globalNamespace_Ref0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type0 = GlobalNamespace.Ref.Type.ALIASING_GET;
      GlobalNamespace.Ref globalNamespace_Ref0 = GlobalNamespace.Ref.createRefForTesting(globalNamespace_Ref_Type0);
      GlobalNamespace.Ref.Type globalNamespace_Ref_Type1 = GlobalNamespace.Ref.Type.SET_FROM_LOCAL;
      GlobalNamespace.Ref globalNamespace_Ref1 = globalNamespace_Ref0.cloneAndReclassify(globalNamespace_Ref_Type1);
      GlobalNamespace.Ref.markTwins(globalNamespace_Ref0, globalNamespace_Ref1);
      assertNotSame(globalNamespace_Ref0, globalNamespace_Ref1);
  }
}
