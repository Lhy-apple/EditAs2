/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:03:07 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Predicate;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.ArrowType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.NamedType;
import com.google.javascript.rhino.jstype.ObjectType;
import com.google.javascript.rhino.jstype.Property;
import com.google.javascript.rhino.jstype.RecordType;
import com.google.javascript.rhino.jstype.RecordTypeBuilder;
import com.google.javascript.rhino.jstype.SimpleSlot;
import com.google.javascript.rhino.jstype.StaticScope;
import java.util.HashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NamedType_ESTest extends NamedType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "com.google.javascript.rhino.jstype.VoidType", "NsCF@:#yb4gTP6_lU>X", 0, 0);
      boolean boolean0 = namedType0.isNamedType();
      assertTrue(boolean0);
      assertEquals("com.google.javascript.rhino.jstype.VoidType", namedType0.getReferenceName());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "com.google.javascript.rhino.jstype.ValueType", "com.google.javascript.rhino.jstype.ValueType", 9, 9);
      String string0 = namedType0.toAnnotationString();
      assertEquals("com.google.javascript.rhino.jstype.ValueType", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "y*N4+1m?.e{2N(}08", "y*N4+1m?.e{2N(}08", 738, 738);
      boolean boolean0 = namedType0.isNominalType();
      assertTrue(boolean0);
      assertEquals("y*N4+1m?.e{2N(}08", namedType0.getReferenceName());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "", "", (-1), (-1));
      jSTypeRegistry0.resolveTypesInScope(namedType0);
      assertEquals("", namedType0.getReferenceName());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      jSTypeRegistry0.forwardDeclareType("o 51~S~#Yv");
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "o 51~S~#Yv", "o 51~S~#Yv", (-21), (-21));
      Node node0 = Node.newString("Not declared as a constructor");
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType(node0, namedType0);
      namedType0.defineDeclaredProperty("o 51~S~#Yv", arrowType0, node0);
      namedType0.resolveInternal(simpleErrorReporter0, namedType0);
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "5 51UYP~S~#Yv", "5 51UYP~S~#Yv", 6, 1);
      Node node0 = Node.newString(0, "Not declared as a type name", 955, 1346);
      namedType0.defineDeclaredProperty("5 51UYP~S~#Yv", (JSType) null, node0);
      boolean boolean0 = namedType0.defineSynthesizedProperty((String) null, namedType0, node0);
      assertTrue(boolean0);
      assertTrue(namedType0.hasReferenceName());
      assertEquals("5 51UYP~S~#Yv", namedType0.getReferenceName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      jSTypeRegistry0.forwardDeclareType("o 51~S~#Yv");
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "o 51~S~#Yv", "o 51~S~#Yv", (-21), (-21));
      namedType0.resolveInternal(simpleErrorReporter0, namedType0);
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      jSTypeRegistry0.setLastGeneration(false);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "null", "null", 1, 1);
      NamedType namedType1 = (NamedType)namedType0.resolveInternal(simpleErrorReporter0, (StaticScope<JSType>) null);
      assertFalse(namedType1.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      jSTypeRegistry0.setLastGeneration(false);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "GXTPt*ROy", "GXTPt*ROy", 5, 5);
      namedType0.resolveInternal(simpleErrorReporter0, namedType0);
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "", "", (-1), (-1));
      namedType0.resolveInternal(simpleErrorReporter0, namedType0);
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "null", "null", (-1432), (-1432));
      Predicate<JSType> predicate0 = (Predicate<JSType>) mock(Predicate.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(predicate0).apply(any(com.google.javascript.rhino.jstype.JSType.class));
      namedType0.setValidator(predicate0);
      JSType jSType0 = namedType0.resolveInternal(simpleErrorReporter0, namedType0);
      assertFalse(jSType0.isUnionType());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      jSTypeRegistry0.forwardDeclareType("o 51~S~#Yv");
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "o 51~S~#Yv", "o 51~S~#Yv", (-21), (-21));
      Predicate<JSType> predicate0 = (Predicate<JSType>) mock(Predicate.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(predicate0).apply(any(com.google.javascript.rhino.jstype.JSType.class));
      namedType0.setValidator(predicate0);
      namedType0.resolveInternal(simpleErrorReporter0, namedType0);
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null, true);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "WNJH;9HizK$[ V0", "WNJH;9HizK$[ V0", 11, 11);
      SimpleSlot simpleSlot0 = new SimpleSlot("Not declared as a constructor", (JSType) null, false);
      // Undeclared exception!
      try { 
        namedType0.getTypedefType((ErrorReporter) null, simpleSlot0, "Not declared as a type name");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.jstype.NamedType", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "wYC<7g'", "wYC<7g'", 391, 391);
      Node node0 = Node.newNumber((double) 0, 25, 8194);
      Property property0 = new Property("Named type with empty name component", namedType0, true, node0);
      JSType jSType0 = namedType0.getTypedefType(simpleErrorReporter0, property0, "wYC<7g'");
      assertNotNull(jSType0);
      assertFalse(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "null", "7~q-Z_3h;#M!\"EU1", (-1), (-1));
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      ObjectType objectType0 = recordType0.getImplicitPrototype();
      namedType0.resolveInternal(simpleErrorReporter0, objectType0);
      // Undeclared exception!
      try { 
        namedType0.setValidator((Predicate<JSType>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.jstype.JSType", e);
      }
  }
}
