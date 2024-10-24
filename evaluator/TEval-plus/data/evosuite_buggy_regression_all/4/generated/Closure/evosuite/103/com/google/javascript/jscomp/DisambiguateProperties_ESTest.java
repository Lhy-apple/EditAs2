/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:19:37 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.collect.Multimap;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ConcreteType;
import com.google.javascript.jscomp.DisambiguateProperties;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.TightenTypes;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.ScriptOrFnNode;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeNative;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.NamedType;
import com.google.javascript.rhino.jstype.RecordType;
import com.google.javascript.rhino.jstype.UnknownType;
import java.util.Collection;
import java.util.HashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DisambiguateProperties_ESTest extends DisambiguateProperties_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("com.google.javascript.jscomp.DisambigaateProperties$JSTypeSystem", "com.google.javascript.jscomp.DisambigaateProperties$JSTypeSystem");
      Node node0 = compiler0.parse(jSSourceFile0);
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      disambiguateProperties0.process(node0, node0);
      disambiguateProperties0.process(node0, node0);
      assertEquals(4095, Node.MAX_COLUMN_NUMBER);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      DisambiguateProperties<ConcreteType> disambiguateProperties0 = DisambiguateProperties.forConcreteTypeSystem(compiler0, tightenTypes0);
      assertNotNull(disambiguateProperties0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, JSType> hashMap0 = new HashMap<String, JSType>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      JSType jSType0 = disambiguateProperties0.getTypeWithProperty("prototype", recordType0);
      assertNull(jSType0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("com.google.javascript.jscomp.DisambiguateProperties$JSTypeSystem", "com.google.javascript.jscomp.DisambiguateProperties$JSTypeSystem");
      ScriptOrFnNode scriptOrFnNode0 = (ScriptOrFnNode)compiler0.parse(jSSourceFile0);
      Node node0 = Node.newString((-2106280314), "&Fgvtf", 27, 9);
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      disambiguateProperties0.process(node0, scriptOrFnNode0);
      assertEquals((-1), scriptOrFnNode0.getBaseLineno());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      Node node0 = Node.newString(64, "TightenTypes pass appears to be stuck in an infinite loop.", 0, 21);
      disambiguateProperties0.process(node0, node0);
      assertEquals(1, Node.SPECIALCALL_EVAL);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      Node node0 = Node.newString(64, "TightenTypes pass appears to be stuck in an infinite loop.", 0, 21);
      Node node1 = Node.newString(18, "TightenTypes pass appears to be stuck in an infinite loop.", 23, 3);
      node0.addChildToFront(node1);
      // Undeclared exception!
      try { 
        disambiguateProperties0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("com.google.javascript.jscomp.DisambigaateProperties$JSTypeSystem", "com.google.javascript.jscomp.DisambigaateProperties$JSTypeSystem");
      Node node0 = compiler0.parse(jSSourceFile0);
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      disambiguateProperties0.process(node0, node0);
      Multimap<String, Collection<JSType>> multimap0 = disambiguateProperties0.getRenamedTypesForTesting();
      assertNotNull(multimap0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      HashMap<String, JSType> hashMap0 = new HashMap<String, JSType>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      JSType jSType0 = jSTypeRegistry0.createOptionalType(recordType0);
      JSType jSType1 = disambiguateProperties0.getTypeWithProperty("TightenTypes pass appears to be stuck in an infinite loop.", jSType0);
      assertNull(jSType1);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      JSTypeNative jSTypeNative0 = JSTypeNative.NUMBER_TYPE;
      JSType jSType0 = jSTypeRegistry0.getNativeType(jSTypeNative0);
      JSType jSType1 = disambiguateProperties0.getTypeWithProperty("[^w$]", jSType0);
      assertNull(jSType1);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      DisambiguateProperties<JSType> disambiguateProperties0 = DisambiguateProperties.forJSTypeSystem(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "", "TightenTypes pass appears to be stuck in an infinite loop.", 64, (-1940));
      JSType jSType0 = namedType0.getReferencedType();
      UnknownType unknownType0 = (UnknownType)disambiguateProperties0.getTypeWithProperty("", jSType0);
      assertFalse(unknownType0.isConstructor());
  }
}
