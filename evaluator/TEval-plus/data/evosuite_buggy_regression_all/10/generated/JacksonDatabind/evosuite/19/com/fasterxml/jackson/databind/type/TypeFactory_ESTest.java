/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:28:07 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.HierarchicType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.type.TypeModifier;
import com.fasterxml.jackson.databind.type.TypeParser;
import java.lang.reflect.Array;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeFactory_ESTest extends TypeFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      JavaType javaType0 = typeFactory0.moreSpecificType(collectionType0, (JavaType) null);
      assertFalse(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeModifier[] typeModifierArray0 = new TypeModifier[0];
      TypeFactory typeFactory1 = new TypeFactory(typeParser0, typeModifierArray0);
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory2 = typeFactory1.withModifier(typeModifier0);
      assertNotSame(typeFactory1, typeFactory2);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      // Undeclared exception!
      try { 
        typeFactory0._fromArrayType((GenericArrayType) null, (TypeBindings) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      objectMapper0.readerFor((JavaType) arrayType0);
      assertTrue(arrayType0.hasGenericTypes());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      assertFalse(javaType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class0, class0);
      Stack<JavaType> stack0 = new Stack<JavaType>();
      stack0.add((JavaType) mapType0);
      typeFactory0._fromParameterizedClass(class0, stack0);
      assertEquals("[[map type; class java.util.HashMap, [map type; class java.util.HashMap, [simple type, class java.lang.Object] -> [simple type, class java.lang.Object]] -> [map type; class java.util.HashMap, [simple type, class java.lang.Object] -> [simple type, class java.lang.Object]]]]", stack0.toString());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ReferenceType> class0 = ReferenceType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class0);
      assertFalse(mapLikeType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      typeFactory0.clearCache();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Object> class0 = Object.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      assertEquals(1, collectionLikeType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<SimpleType> class0 = SimpleType.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class0);
      assertFalse(collectionLikeType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      // Undeclared exception!
      try { 
        typeFactory0.constructType((TypeReference<?>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      Class<?> class1 = mapType0.getParameterSource();
      TypeBindings typeBindings0 = new TypeBindings(typeFactory0, mapType0);
      JavaType javaType0 = typeFactory0._constructType(class1, typeBindings0);
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructCollectionLikeType(class0, (JavaType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.CollectionLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<CollectionType> class0 = CollectionType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      assertFalse(mapLikeType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      try { 
        typeFactory0.constructFromCanonical("b=h\"WEk");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Failed to parse type 'b=h\"WEk' (remaining: ''): Can not locate class 'b=h\"WEk', problem: Class 'b=h\"WEk.class' should be in target project, but could not be found!
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<JsonToken> class0 = JsonToken.class;
      Class<Module>[] classArray0 = (Class<Module>[]) Array.newInstance(Class.class, 8);
      // Undeclared exception!
      try { 
        typeFactory0.constructParametricType(class0, classArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.ClassKey", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructMapLikeType(class0, (JavaType) null, (JavaType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.MapLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0._unknownType();
      ArrayType arrayType0 = typeFactory0.constructArrayType(javaType0);
      assertTrue(arrayType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeFactory typeFactory1 = typeFactory0.withModifier((TypeModifier) null);
      assertNotSame(typeFactory1, typeFactory0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      // Undeclared exception!
      try { 
        TypeFactory.rawClass((Type) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unrecognized Type: [null]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<SimpleType> class0 = SimpleType.class;
      Class<?> class1 = TypeFactory.rawClass(class0);
      assertFalse(class1.isAnnotation());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      Class<Object> class0 = Object.class;
      JavaType javaType0 = typeFactory0.constructSpecializedType(simpleType0, class0);
      assertFalse(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JavaType javaType0 = typeFactory0.constructSpecializedType(simpleType0, class0);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0._unknownType();
      Class<HashMap> class0 = HashMap.class;
      JavaType javaType1 = typeFactory0.constructSpecializedType(javaType0, class0);
      assertFalse(javaType1.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      JavaType javaType0 = typeFactory0.constructSpecializedType(simpleType0, class0);
      assertFalse(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      Class<HashMap> class0 = HashMap.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructSpecializedType(simpleType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class java.lang.Class not subtype of [simple type, class int]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      Class<ReferenceType> class1 = ReferenceType.class;
      // Undeclared exception!
      try { 
        typeFactory0.findTypeParameters((JavaType) collectionType0, (Class<?>) class1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class java.util.ArrayList is not a subtype of com.fasterxml.jackson.databind.type.ReferenceType
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters((JavaType) simpleType0, (Class<?>) null);
      assertNull(javaTypeArray0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      Class<Object> class1 = Object.class;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters((JavaType) collectionType0, (Class<?>) class1);
      assertNull(javaTypeArray0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0._unknownType();
      JavaType javaType1 = typeFactory0.moreSpecificType(javaType0, javaType0);
      assertFalse(javaType1.isCollectionLikeType());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0.moreSpecificType((JavaType) null, (JavaType) null);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      SimpleType simpleType1 = TypeFactory.CORE_TYPE_INT;
      JavaType javaType0 = typeFactory0.moreSpecificType(simpleType0, simpleType1);
      assertFalse(javaType0.isAbstract());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MapType> class0 = MapType.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (Class<?>) class0);
      assertFalse(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      // Undeclared exception!
      try { 
        typeFactory0.constructType((Type) null, (Class<?>) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unrecognized Type: [null]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      JavaType javaType0 = typeFactory0.constructType((Type) simpleType0, (JavaType) simpleType0);
      assertFalse(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayType> class0 = ArrayType.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (JavaType) null);
      assertFalse(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeModifier[] typeModifierArray0 = new TypeModifier[0];
      TypeFactory typeFactory1 = new TypeFactory(typeParser0, typeModifierArray0);
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory1.constructCollectionType(class0, class0);
      assertEquals(1, collectionType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      doReturn((JavaType) null).when(typeModifier0).modifyType(any(com.fasterxml.jackson.databind.JavaType.class) , any(java.lang.reflect.Type.class) , any(com.fasterxml.jackson.databind.type.TypeBindings.class) , any(com.fasterxml.jackson.databind.type.TypeFactory.class));
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      Class<ArrayList> class0 = ArrayList.class;
      Class<Integer> class1 = Integer.class;
      // Undeclared exception!
      try { 
        typeFactory1.constructCollectionType(class0, class1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.CollectionLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<JsonToken> class0 = JsonToken.class;
      Class<Module>[] classArray0 = (Class<Module>[]) Array.newInstance(Class.class, 0);
      JavaType javaType0 = typeFactory0.constructParametricType(class0, classArray0);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      JavaType[] javaTypeArray0 = new JavaType[2];
      // Undeclared exception!
      try { 
        typeFactory0.constructParametricType(class0, javaTypeArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.MapLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      // Undeclared exception!
      try { 
        typeFactory0.constructParametricType(class0, javaTypeArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Need exactly 2 parameter types for Map types (java.util.HashMap)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      Class<CollectionLikeType> class1 = CollectionLikeType.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class0, class1, javaTypeArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.CollectionLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      // Undeclared exception!
      try { 
        typeFactory0.constructParametricType(class0, javaTypeArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Need exactly 1 parameter type for Collection types (java.util.ArrayList)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      Class<Integer> class1 = Integer.TYPE;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      assertFalse(collectionType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerFor((JavaType) simpleType0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<JsonDeserializer> class0 = JsonDeserializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      Class<LinkedList> class1 = LinkedList.class;
      typeFactory0.constructCollectionType((Class<? extends Collection>) class1, (JavaType) simpleType0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayType arrayType0 = typeFactory0.constructArrayType(class1);
      assertTrue(arrayType0.hasGenericTypes());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      Class<JsonToken> class1 = JsonToken.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      assertFalse(collectionType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<JsonSerializer> class1 = JsonSerializer.class;
      ObjectWriter objectWriter0 = objectMapper0.writerFor(class1);
      assertTrue(objectWriter0.hasPrefetchedSerializer());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      Stack<JavaType> stack0 = new Stack<JavaType>();
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, stack0);
      assertFalse(javaType0.isCollectionLikeType());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Stack<JavaType> stack0 = new Stack<JavaType>();
      Class<JsonToken> class0 = JsonToken.class;
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, stack0);
      assertEquals(0, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      Stack<JavaType> stack0 = new Stack<JavaType>();
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, stack0);
      assertFalse(javaType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      Stack<JavaType> stack0 = new Stack<JavaType>();
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Integer> class1 = Integer.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class1);
      stack0.add((JavaType) arrayType0);
      JavaType javaType0 = objectMapper0.constructType(class0);
      stack0.add(javaType0);
      typeFactory0._fromParameterizedClass(class0, stack0);
      assertEquals("[[array type, component type: [simple type, class java.lang.Integer]], [map type; class java.util.HashMap, [simple type, class java.lang.Object] -> [simple type, class java.lang.Object]]]", stack0.toString());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Object> class0 = Object.class;
      LinkedList<JavaType> linkedList0 = new LinkedList<JavaType>();
      JavaType javaType0 = typeFactory0._fromParameterizedClass(class0, linkedList0);
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      Stack<JavaType> stack0 = new Stack<JavaType>();
      stack0.add((JavaType) null);
      // Undeclared exception!
      try { 
        typeFactory0._fromParameterizedClass(class0, stack0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.CollectionLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MapLikeType> class0 = MapLikeType.class;
      Stack<JavaType> stack0 = new Stack<JavaType>();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      stack0.add((JavaType) simpleType0);
      // Undeclared exception!
      try { 
        typeFactory0._fromParameterizedClass(class0, stack0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Parameter type mismatch for com.fasterxml.jackson.databind.type.MapLikeType (and target com.fasterxml.jackson.databind.type.MapLikeType): expected 0 parameters, was given 1
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      Class<JsonDeserializer> class1 = JsonDeserializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class1);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectMapper objectMapper1 = objectMapper0.setTypeFactory(typeFactory0);
      ObjectReader objectReader0 = objectMapper1.readerFor((JavaType) simpleType0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0._resolveVariableViaSubTypes((HierarchicType) null, "VW7wb0c", (TypeBindings) null);
      assertFalse(javaType0.hasValueHandler());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      Class<ReferenceType> class1 = ReferenceType.class;
      HierarchicType hierarchicType0 = typeFactory0._findSuperInterfaceChain(class0, class1);
      assertNull(hierarchicType0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      Class<Object> class1 = Object.class;
      HierarchicType hierarchicType0 = typeFactory0._findSuperInterfaceChain(class0, class1);
      assertFalse(hierarchicType0.isGeneric());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      HierarchicType hierarchicType0 = typeFactory0._cachedHashMapType;
      // Undeclared exception!
      try { 
        typeFactory0._arrayListSuperInterfaceChain(hierarchicType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }
}