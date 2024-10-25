/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:22:07 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.collect.Multimap;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ConcreteType;
import com.google.javascript.jscomp.DisambiguateProperties;
import com.google.javascript.jscomp.TightenTypes;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.NamedType;
import com.google.javascript.rhino.jstype.RecordType;
import java.util.Collection;
import java.util.HashMap;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DisambiguateProperties_ESTest extends DisambiguateProperties_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      Node node0 = new Node(33, (-2362), 614);
      Node node1 = Node.newString("7Rx STG'", (-1), (-2362));
      node0.addChildToBack(node1);
      disambiguateProperties0.process(node1, node0);
      assertEquals(13, Node.CASES_PROP);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, tightenTypes0);
      // Undeclared exception!
      try { 
        disambiguateProperties0.process((Node) null, (Node) null);
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
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      Node node0 = new Node(33, (-2362), 614);
      Node node1 = Node.newString("7Rx STG'", 990, (-2362));
      node0.addChildToBack(node1);
      disambiguateProperties0.process(node0, node0);
      assertEquals(45, Node.IS_VAR_ARGS_PARAM);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      Node node0 = new Node(64, 64, 64);
      disambiguateProperties0.process(node0, node0);
      assertEquals(5, Node.FUNCTION_PROP);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      Multimap<String, Collection<JSType>> multimap0 = disambiguateProperties0.getRenamedTypesForTesting();
      assertNotNull(multimap0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      HashMap<String, JSType> hashMap0 = new HashMap<String, JSType>();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      JSType jSType0 = disambiguateProperties0.getTypeWithProperty("prototype", recordType0);
      assertNull(jSType0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      HashMap<String, JSType> hashMap0 = new HashMap<String, JSType>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      JSType jSType0 = jSTypeRegistry0.createNullableType(recordType0);
      JSType jSType1 = disambiguateProperties0.getTypeWithProperty("v%XF.gU<g.}I\".", jSType0);
      assertNull(jSType1);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "G9lCTKK7+!`;v9bg,", "@define variable {0} assignment must be global", (-39), (-39));
      NamedType namedType1 = (NamedType)disambiguateProperties0.getTypeWithProperty("@define variable {0} assignment must be global", namedType0);
      assertFalse(namedType1.isNoObjectType());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      HashMap<String, JSType> hashMap0 = new HashMap<String, JSType>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      JSType jSType0 = disambiguateProperties0.getTypeWithProperty("Not declared as a constructor", recordType0);
      assertNull(jSType0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ConcreteType concreteType0 = ConcreteType.ALL;
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, tightenTypes0);
      ConcreteType concreteType1 = disambiguateProperties0.getTypeWithProperty("TightenTypes pass appears to be stuck in an infinite loop.", concreteType0);
      assertNull(concreteType1);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, tightenTypes0);
      Vector<ConcreteType> vector0 = new Vector<ConcreteType>(64);
      ConcreteType concreteType0 = ConcreteType.createForTypes(vector0);
      ConcreteType concreteType1 = disambiguateProperties0.getTypeWithProperty("TightenTypes pass appears to be stuck in an infinite loop.", concreteType0);
      assertNotNull(concreteType1);
  }
}
