/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:09:21 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.ClassStack;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.type.TypeModifier;
import com.fasterxml.jackson.databind.type.TypeParser;
import java.lang.reflect.Array;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Properties;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeFactory_ESTest extends TypeFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Object> class0 = Object.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructType((Type) null, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unrecognized Type: [null]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MapperFeature> class0 = MapperFeature.class;
      JavaType javaType0 = typeFactory0.constructSimpleType(class0, class0, (JavaType[]) null);
      assertTrue(javaType0.isEnumType());
      assertNotNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      JavaType javaType0 = typeFactory0.constructGeneralizedType(collectionType0, class0);
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeFactory typeFactory1 = new TypeFactory(typeParser0, (TypeModifier[]) null);
      assertFalse(typeFactory1.equals((Object)typeFactory0));
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Object> class0 = Object.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Integer> class0 = Integer.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      JavaType javaType0 = typeFactory0.constructParametrizedType(class0, class0, javaTypeArray0);
      assertFalse(javaType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      assertFalse(javaType0.isFinal());
      assertNotNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MapperFeature> class0 = MapperFeature.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class0, class0, (Class<?>[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      typeFactory0.clearCache();
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      assertTrue(collectionLikeType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashSet> class0 = HashSet.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      MapType mapType0 = new MapType(collectionType0, collectionType0, collectionType0);
      TypeBindings typeBindings0 = mapType0._bindings;
      JavaType[] javaTypeArray0 = new JavaType[5];
      JavaType javaType0 = typeFactory0._constructSimple(class0, typeBindings0, collectionType0, javaTypeArray0);
      assertEquals(1, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class0);
      JavaType javaType0 = typeFactory0.constructSpecializedType(collectionLikeType0, class0);
      assertFalse(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapperFeature> class0 = MapperFeature.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      assertFalse(arrayType0.isMapLikeType());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
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
  public void test14()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructRawMapType(class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Properties with 2 type parameters: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<CollectionType> class0 = CollectionType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      assertTrue(mapLikeType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0.constructFromCanonical("com.fasterxml.jackson.databind.type.TypeFactory");
      assertFalse(javaType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassLoader classLoader0 = ClassLoader.getSystemClassLoader();
      TypeFactory typeFactory1 = typeFactory0.withClassLoader(classLoader0);
      try { 
        typeFactory1.findClass("[null]");
        fail("Expecting exception: ClassNotFoundException");
      
      } catch(ClassNotFoundException e) {
         //
         // [null]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapType> class0 = MapType.class;
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(class0, class0, typeBindings0);
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapperFeature> class0 = MapperFeature.class;
      SimpleType simpleType0 = new SimpleType(class0);
      JavaType javaType0 = typeFactory0.constructReferenceType(class0, simpleType0);
      assertFalse(javaType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      ArrayType arrayType0 = typeFactory0.constructArrayType((JavaType) simpleType0);
      assertFalse(arrayType0.isAbstract());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      TypeModifier typeModifier1 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory2 = typeFactory1.withModifier(typeModifier1);
      assertNotSame(typeFactory1, typeFactory2);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeFactory typeFactory1 = typeFactory0.withModifier((TypeModifier) null);
      assertNotSame(typeFactory1, typeFactory0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ReferenceType> class0 = ReferenceType.class;
      Class<MapperFeature> class1 = MapperFeature.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class1);
      Class<?> class2 = TypeFactory.rawClass(collectionLikeType0);
      assertEquals(1, class2.getModifiers());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      Class<?> class1 = TypeFactory.rawClass(class0);
      assertFalse(class1.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("char");
      assertEquals("char", class0.toString());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("int");
      assertTrue(class0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("long");
      JavaType javaType0 = typeFactory0._findWellKnownSimple(class0);
      assertNotNull(javaType0);
      assertTrue(javaType0.isPrimitive());
      assertEquals("long", class0.toString());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("float");
      assertEquals("float", class0.toString());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("double");
      assertEquals("double", class0.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("boolean");
      assertEquals("boolean", class0.toString());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("byte");
      assertEquals("byte", class0.toString());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("short");
      assertEquals("short", class0.toString());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("void");
      typeFactory0.constructCollectionLikeType(class0, class0);
      assertEquals("void", class0.toString());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = TypeFactory.unknownType();
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      JavaType javaType1 = typeFactory0.constructSpecializedType(javaType0, class0);
      assertTrue(javaType0.isJavaLangObject());
      assertTrue(javaType1.isEnumType());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructSpecializedType(simpleType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.type.ResolvedRecursiveType not subtype of [simple type, class int]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_COMPARABLE;
      Class<String> class0 = String.class;
      JavaType javaType0 = typeFactory0.constructSpecializedType(simpleType0, class0);
      assertFalse(javaType0.isJavaLangObject());
      assertFalse(javaType0.isAbstract());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      ClassStack classStack0 = new ClassStack(class0);
      Class<ReferenceType> class1 = ReferenceType.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      Properties properties0 = new Properties();
      MapLikeType mapLikeType0 = new MapLikeType(class1, typeBindings0, simpleType0, (JavaType[]) null, simpleType0, simpleType0, properties0, classStack0, true);
      ReferenceType referenceType0 = new ReferenceType(mapLikeType0, simpleType0);
      // Undeclared exception!
      try { 
        typeFactory0.constructGeneralizedType(referenceType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.type.ResolvedRecursiveType not a super-type of [reference type, class com.fasterxml.jackson.databind.type.ReferenceType<java.lang.Enum<[simple type, class java.lang.Enum]>]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      ClassStack classStack0 = new ClassStack(class0);
      Class<ReferenceType> class1 = ReferenceType.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      SimpleType simpleType1 = new SimpleType(class0, typeBindings0, simpleType0, (JavaType[]) null, classStack0, typeBindings0, true);
      Properties properties0 = new Properties();
      MapLikeType mapLikeType0 = new MapLikeType(class1, typeBindings0, simpleType1, (JavaType[]) null, simpleType1, simpleType1, properties0, classStack0, true);
      ReferenceType referenceType0 = new ReferenceType(mapLikeType0, simpleType1);
      JavaType javaType0 = typeFactory0.constructGeneralizedType(referenceType0, class0);
      assertFalse(javaType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      Class<Object> class0 = Object.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructGeneralizedType(simpleType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Internal error: class java.lang.Object not included as super-type for [simple type, class java.lang.Class]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<CollectionType> class0 = CollectionType.class;
      Class<ObjectMapper.DefaultTyping> class1 = ObjectMapper.DefaultTyping.class;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(class0, class1);
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Class<MapperFeature> class0 = MapperFeature.class;
      JavaType javaType0 = typeFactory0._newSimpleType(class0, typeBindings0, (JavaType) null, (JavaType[]) null);
      JavaType javaType1 = typeFactory0.moreSpecificType(javaType0, (JavaType) null);
      assertSame(javaType0, javaType1);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0.moreSpecificType((JavaType) null, (JavaType) null);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      JavaType javaType0 = typeFactory0.moreSpecificType(simpleType0, simpleType0);
      assertEquals(0, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      JavaType javaType0 = typeFactory0._unknownType();
      JavaType javaType1 = typeFactory0.moreSpecificType(javaType0, simpleType0);
      assertTrue(javaType1.isFinal());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Class<MapperFeature> class0 = MapperFeature.class;
      JavaType javaType0 = typeFactory0._newSimpleType(class0, typeBindings0, simpleType0, (JavaType[]) null);
      JavaType javaType1 = typeFactory0.moreSpecificType(javaType0, simpleType0);
      assertTrue(javaType1.isEnumType());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<CollectionType> class0 = CollectionType.class;
      Class<DeserializationFeature> class1 = DeserializationFeature.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructMapType((Class<? extends Map>) null, class0, class1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings$TypeParamStash", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<JsonDeserializer> class0 = JsonDeserializer.class;
      Class<Properties> class1 = Properties.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructMapType(class1, class1, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Properties with 2 type parameters: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Properties> class0 = Properties.class;
      Class<MapperFeature> class1 = MapperFeature.class;
      Class<ArrayType> class2 = ArrayType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class1, class2);
      assertFalse(mapLikeType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Properties> class0 = Properties.class;
      Class<Integer>[] classArray0 = (Class<Integer>[]) Array.newInstance(Class.class, 1);
      Class<Integer> class1 = Integer.class;
      classArray0[0] = class1;
      // Undeclared exception!
      try { 
        typeFactory0.constructParametricType(class0, classArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Properties with 1 type parameter: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      assertNotNull(javaType0);
      assertFalse(javaType0.isAbstract());
      assertFalse(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<SimpleType> class0 = SimpleType.class;
      ClassStack classStack0 = new ClassStack(class0);
      GenericArrayType genericArrayType0 = mock(GenericArrayType.class, new ViolatedAssumptionAnswer());
      doReturn(class0).when(genericArrayType0).getGenericComponentType();
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      JavaType javaType0 = typeFactory0._fromAny(classStack0, genericArrayType0, typeBindings0);
      assertFalse(javaType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      doReturn((JavaType) null).when(typeModifier0).modifyType(any(com.fasterxml.jackson.databind.JavaType.class) , any(java.lang.reflect.Type.class) , any(com.fasterxml.jackson.databind.type.TypeBindings.class) , any(com.fasterxml.jackson.databind.type.TypeFactory.class));
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      CollectionLikeType collectionLikeType0 = typeFactory1.constructCollectionLikeType(class0, class0);
      assertFalse(collectionLikeType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      Class<ArrayList> class0 = ArrayList.class;
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      DeserializationFeature deserializationFeature0 = DeserializationFeature.ACCEPT_FLOAT_AS_INT;
      DeserializationFeature[] deserializationFeatureArray0 = new DeserializationFeature[3];
      deserializationFeatureArray0[0] = deserializationFeature0;
      deserializationFeatureArray0[1] = deserializationFeature0;
      deserializationFeatureArray0[2] = deserializationFeatureArray0[0];
      ObjectReader objectReader0 = objectMapper0.reader(deserializationFeature0, deserializationFeatureArray0);
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      TypeModifier[] typeModifierArray0 = new TypeModifier[0];
      ClassLoader classLoader0 = ClassLoader.getSystemClassLoader();
      TypeFactory typeFactory1 = new TypeFactory(typeParser0, typeModifierArray0, classLoader0);
      CollectionType collectionType0 = typeFactory1.constructRawCollectionType(class0);
      assertFalse(collectionType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<ReferenceType> class0 = ReferenceType.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ParameterizedType parameterizedType0 = mock(ParameterizedType.class, new ViolatedAssumptionAnswer());
      doReturn((Type[]) null).when(parameterizedType0).getActualTypeArguments();
      doReturn((Type) null).when(parameterizedType0).getRawType();
      // Undeclared exception!
      try { 
        typeFactory0._fromParamType((ClassStack) null, parameterizedType0, typeBindings0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }
}