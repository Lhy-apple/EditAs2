/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:27:15 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.IndexedType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.NamedType;
import com.google.javascript.rhino.jstype.NoObjectType;
import com.google.javascript.rhino.jstype.NoResolvedType;
import com.google.javascript.rhino.jstype.ObjectType;
import com.google.javascript.rhino.jstype.RecordType;
import com.google.javascript.rhino.jstype.RecordTypeBuilder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class RecordType_ESTest extends RecordType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      hashMap0.put("Not declared as a type name", (RecordTypeBuilder.RecordProperty) null);
      RecordType recordType0 = null;
      try {
        recordType0 = new RecordType(jSTypeRegistry0, hashMap0, true);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // RecordProperty associated with a property should not be null!
         //
         verifyException("com.google.javascript.rhino.jstype.RecordType", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, true);
      Node node0 = new Node(0, 1, 0);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(recordType0, node0);
      hashMap0.put("Not declared as a constructor", recordTypeBuilder_RecordProperty0);
      RecordType recordType1 = new RecordType(jSTypeRegistry0, hashMap0, true);
      JSType jSType0 = recordType1.getGreatestSubtypeHelper(recordType1);
      assertTrue(jSType0.equals((Object)recordType1));
      assertFalse(recordType1.equals((Object)recordType0));
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      boolean boolean0 = recordType0.isSynthetic();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, false);
      boolean boolean0 = recordType0.isSynthetic();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, true);
      Node node0 = new Node(0, 1, 0);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(recordType0, node0);
      hashMap0.put("Not declared as a constructor", recordTypeBuilder_RecordProperty0);
      hashMap0.put("first new child has existing sibling", recordTypeBuilder_RecordProperty0);
      RecordType recordType1 = new RecordType(jSTypeRegistry0, hashMap0, true);
      RecordType recordType2 = jSTypeRegistry0.createRecordType(hashMap0);
      assertTrue(recordType2.equals((Object)recordType1));
      assertFalse(recordType2.equals((Object)recordType0));
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "", "", (-686), (-686));
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(namedType0, (Node) null);
      hashMap0.put("Unknown class name", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      JSType jSType0 = recordType0.resolveInternal(simpleErrorReporter0, noResolvedType0);
      RecordType recordType1 = new RecordType(jSTypeRegistry0, hashMap0, false);
      boolean boolean0 = JSType.isEquivalent(recordType1, jSType0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      Node node0 = Node.newNumber((double) 0);
      boolean boolean0 = recordType0.defineProperty("com.google.javascript.rhino.jstype.NamedType$PropertyContinuation", (JSType) null, true, node0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      JSType jSType0 = recordType0.getGreatestSubtypeHelper(noObjectType0);
      assertEquals(1, JSType.ENUMDECL);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, true);
      Node node0 = new Node(0, 1, 1);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(recordType0, node0);
      hashMap0.put("Not declared as a constructor", recordTypeBuilder_RecordProperty0);
      RecordType recordType1 = new RecordType(jSTypeRegistry0, hashMap0, true);
      JSType jSType0 = recordType1.getGreatestSubtypeHelper(recordType0);
      assertFalse(recordType1.equals((Object)recordType0));
      assertTrue(jSType0.equals((Object)recordType1));
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "", "", (-686), (-686));
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(namedType0, (Node) null);
      hashMap0.put("Unknown class name", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, false);
      recordType0.resolveInternal(simpleErrorReporter0, namedType0);
      assertTrue(namedType0.isResolved());
      
      RecordType recordType1 = new RecordType(jSTypeRegistry0, hashMap0);
      recordType0.getGreatestSubtypeHelper(recordType1);
      assertFalse(recordType0.equals((Object)recordType1));
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, true);
      Node node0 = new Node(0, 1, 1);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(recordType0, node0);
      hashMap0.put("Not declared as a constructor", recordTypeBuilder_RecordProperty0);
      RecordType recordType1 = new RecordType(jSTypeRegistry0, hashMap0, true);
      JSType jSType0 = recordType0.getGreatestSubtypeHelper(recordType1);
      assertTrue(jSType0.equals((Object)recordType1));
      assertFalse(recordType1.equals((Object)recordType0));
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "", "", (-686), (-686));
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(namedType0, (Node) null);
      hashMap0.put("Not declared as a type name", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, true);
      NoObjectType noObjectType0 = (NoObjectType)recordType0.getGreatestSubtypeHelper(namedType0);
      assertTrue(noObjectType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, true);
      Node node0 = new Node(0, 1, 1);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(recordType0, node0);
      hashMap0.put("Not declared as a constructor", recordTypeBuilder_RecordProperty0);
      RecordType recordType1 = new RecordType(jSTypeRegistry0, hashMap0, true);
      recordType1.canTestForShallowEqualityWith(recordType0);
      assertFalse(recordType1.equals((Object)recordType0));
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, false);
      IndexedType indexedType0 = new IndexedType(jSTypeRegistry0, recordType0, recordType0);
      JSType[] jSTypeArray0 = new JSType[3];
      jSTypeArray0[2] = (JSType) recordType0;
      FunctionType functionType0 = jSTypeRegistry0.createFunctionTypeWithVarArgs((JSType) indexedType0, jSTypeArray0);
      assertFalse(functionType0.isAllType());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "", "", (-686), (-686));
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(namedType0, (Node) null);
      hashMap0.put("Not declared as a type name", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, true);
      boolean boolean0 = RecordType.isSubtype((ObjectType) recordType0, recordType0);
      assertTrue(recordType0.hasCachedValues());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "}", "}", (-698), (-698));
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(namedType0, (Node) null);
      hashMap0.put("Unknown class name", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, false);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      boolean boolean0 = RecordType.isSubtype((ObjectType) noObjectType0, recordType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      ArrayList<JSType> arrayList0 = new ArrayList<JSType>();
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType((JSType) null, (List<JSType>) arrayList0);
      Node node0 = Node.newNumber((double) (-698));
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(functionType0, node0);
      hashMap0.put("TxnOuLl", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      boolean boolean0 = RecordType.isSubtype((ObjectType) noObjectType0, recordType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      ArrayList<JSType> arrayList0 = new ArrayList<JSType>();
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType((JSType) null, (List<JSType>) arrayList0);
      Node node0 = Node.newNumber((double) (-698));
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(functionType0, node0);
      hashMap0.put("TxnOuLl", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      JSType jSType0 = recordType0.forceResolve(simpleErrorReporter0, recordType0);
      assertFalse(jSType0.isNullType());
  }
}