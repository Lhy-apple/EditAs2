/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:02:21 GMT 2023
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
import com.google.javascript.rhino.jstype.EnumElementType;
import com.google.javascript.rhino.jstype.EnumType;
import com.google.javascript.rhino.jstype.ErrorFunctionType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.NamedType;
import com.google.javascript.rhino.jstype.NoObjectType;
import com.google.javascript.rhino.jstype.Property;
import com.google.javascript.rhino.jstype.RecordType;
import com.google.javascript.rhino.jstype.RecordTypeBuilder;
import java.util.HashMap;
import java.util.function.Function;
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
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "com.google.javascript.rhino.jstype.NamedType$PropertyContinuation", "Named type with empty name component", (-1), (-1255));
      boolean boolean0 = namedType0.isNamedType();
      assertTrue(boolean0);
      assertEquals("com.google.javascript.rhino.jstype.NamedType$PropertyContinuation", namedType0.getReferenceName());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "T9", "XPRJDL:v?Jw`WYZ]z", (-391), 0);
      String string0 = namedType0.toStringHelper(true);
      assertEquals("T9", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "8", "8", 2545, 2545);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "com.google.javascript.rhino.SourcePosition");
      errorFunctionType0.setImplicitPrototype(namedType0);
      namedType0.setReferencedType(errorFunctionType0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      namedType0.resolveInternal(simpleErrorReporter0, recordType0);
      assertFalse(namedType0.isNativeObjectType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "e<JH7l/N`Y_'Mi", "e<JH7l/N`Y_'Mi", (-1639), (-1639));
      boolean boolean0 = namedType0.isNominalType();
      assertTrue(boolean0);
      assertEquals("e<JH7l/N`Y_'Mi", namedType0.getReferenceName());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "com.google.javascript.rhino.jstype.NamedType$PropertyContinuation", "Named type with empty name component", (-1), (-1255));
      boolean boolean0 = namedType0.hasReferenceName();
      assertEquals("com.google.javascript.rhino.jstype.NamedType$PropertyContinuation", namedType0.getReferenceName());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Recorded bad position information\nstart-char: ", "Recorded bad position information\nstart-char: ", 2373, 2373);
      namedType0.hashCode();
      assertEquals("Recorded bad position information\nstart-char: ", namedType0.getReferenceName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "", "", 4282, 4282);
      String string0 = namedType0.getReferenceName();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, false);
      jSTypeRegistry0.declareType("Not declared as a type name", recordType0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Not declared as a type name", "Not declared as a constructor", 0, 0);
      Node node0 = Node.newString("Not declared as a type name");
      namedType0.defineSynthesizedProperty("Not declared as a constructor", recordType0, node0);
      namedType0.resolveInternal(simpleErrorReporter0, recordType0);
      assertTrue(namedType0.isRecordType());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "~", "~", (-2719), (-2719));
      Node node0 = new Node((-2719), (-1255), 0);
      namedType0.defineProperty("Not declared as a constructor", namedType0, false, node0);
      boolean boolean0 = namedType0.defineProperty("~", namedType0, true, node0);
      assertEquals("~", namedType0.getReferenceName());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, false);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "void", "Unknown class name", 0, 1);
      namedType0.resolveInternal(simpleErrorReporter0, recordType0);
      assertEquals("void", namedType0.getReferenceName());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      jSTypeRegistry0.forwardDeclareType("8");
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "8", "8", 2545, 2545);
      namedType0.resolveInternal(simpleErrorReporter0, namedType0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      Node node0 = Node.newString("com.google.javascript.rhino.jstype.PropertyMap$1", 0, 1);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(namedType0, node0);
      hashMap0.put("8", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      // Undeclared exception!
      try { 
        namedType0.resolveInternal((ErrorReporter) null, recordType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.jstype.NamedType", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, false);
      jSTypeRegistry0.declareType("Not declared as a type name", recordType0);
      jSTypeRegistry0.setLastGeneration(false);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Not declared as a type name", "Not declared as a constructor", 0, 0);
      namedType0.resolveInternal(simpleErrorReporter0, recordType0);
      assertTrue(namedType0.isRecordType());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, false);
      jSTypeRegistry0.setLastGeneration(false);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Not declared as a type name", "Not declared as a constructor", 0, 0);
      namedType0.resolveInternal(simpleErrorReporter0, recordType0);
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "8", "8", 2545, 2545);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      Node node0 = Node.newString("com.google.javascript.rhino.jstype.PropertyMap$1", 0, 1);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(namedType0, node0);
      hashMap0.put("8", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      // Undeclared exception!
      try { 
        namedType0.resolveInternal((ErrorReporter) null, recordType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.jstype.NamedType", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "L4O=SBu3");
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(errorFunctionType0, (Node) null);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      hashMap0.put("6!wqx{", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "6!wqx{", "Not declared as a constructor", 50, 40);
      namedType0.resolveInternal(simpleErrorReporter0, recordType0);
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "", "", 4282, 4282);
      namedType0.resolveInternal(simpleErrorReporter0, namedType0);
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "DS", "DS", 1764, 1764);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty((JSType) null, (Node) null);
      Function<Object, RecordTypeBuilder.RecordProperty> function0 = (Function<Object, RecordTypeBuilder.RecordProperty>) mock(Function.class, new ViolatedAssumptionAnswer());
      doReturn(recordTypeBuilder_RecordProperty0).when(function0).apply(any());
      hashMap0.computeIfAbsent("DS", function0);
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      namedType0.resolveInternal(simpleErrorReporter0, recordType0);
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Not declared as a type name", "Unknown class name", 2, 2);
      EnumType enumType0 = jSTypeRegistry0.createEnumType("Not declared as a constructor", (Node) null, namedType0);
      EnumElementType enumElementType0 = enumType0.getElementsType();
      jSTypeRegistry0.declareType("Not declared as a type name", enumElementType0);
      NamedType namedType1 = new NamedType(jSTypeRegistry0, "Not declared as a type name", "1r5mg60SUmR)Ky__ty", 7, (-3110));
      namedType1.resolveInternal(simpleErrorReporter0, namedType0);
      assertTrue(namedType1.isEnumElementType());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      jSTypeRegistry0.forwardDeclareType("L4O=SBu3");
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "L4O=SBu3", "L4O=SBu3", 1386, 1386);
      Predicate<JSType> predicate0 = (Predicate<JSType>) mock(Predicate.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(predicate0).apply(any(com.google.javascript.rhino.jstype.JSType.class));
      namedType0.setValidator(predicate0);
      namedType0.resolveInternal(simpleErrorReporter0, namedType0);
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "IoX`*O3!Id", (String) null, 81, 81);
      Property property0 = new Property("3tx", (JSType) null, true, (Node) null);
      namedType0.getTypedefType(simpleErrorReporter0, property0, "Not declared as a constructor");
      assertTrue(namedType0.isResolved());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "T9", "XPRJDL:v?Jw`WYZ]z", (-391), 0);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      namedType0.resolveInternal(simpleErrorReporter0, noObjectType0);
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
