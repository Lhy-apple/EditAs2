/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:51:26 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.LongNode;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.ClassStack;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.type.TypeModifier;
import com.fasterxml.jackson.databind.util.LRUMap;
import java.lang.reflect.Array;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collection;
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
      Class<CollectionType> class0 = CollectionType.class;
      Class<MapType> class1 = MapType.class;
      JavaType javaType0 = typeFactory0.constructSimpleType(class0, class1, (JavaType[]) null);
      assertNotNull(javaType0);
      assertTrue(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      assertEquals(1, collectionType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Object> class0 = Object.class;
      // Undeclared exception!
      try { 
        objectMapper0.readerFor(class0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // com.fasterxml.jackson.databind.type.ReferenceType cannot be cast to com.fasterxml.jackson.databind.type.CollectionType
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MapType> class0 = MapType.class;
      Class<Object>[] classArray0 = (Class<Object>[]) Array.newInstance(Class.class, 0);
      JavaType javaType0 = typeFactory0.constructParametricType(class0, classArray0);
      assertTrue(javaType0.isReferenceType());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapLikeType> class0 = MapLikeType.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      JavaType javaType1 = typeFactory0.constructGeneralizedType(javaType0, class0);
      assertFalse(javaType1.hasContentType());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapLikeType> class0 = MapLikeType.class;
      Class<MapperFeature> class1 = MapperFeature.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class1, class1);
      assertFalse(mapLikeType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<String> class0 = String.class;
      Class<Integer> class1 = Integer.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class0, class1, javaTypeArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.lang.String with 1 type parameter: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ObjectIdResolver> class0 = ObjectIdResolver.class;
      Class<MapType>[] classArray0 = (Class<MapType>[]) Array.newInstance(Class.class, 1);
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class0, class0, classArray0);
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
      TypeFactory typeFactory0 = TypeFactory.instance;
      GenericArrayType genericArrayType0 = mock(GenericArrayType.class, new ViolatedAssumptionAnswer());
      doReturn((Type) null).when(genericArrayType0).getGenericComponentType();
      // Undeclared exception!
      try { 
        typeFactory0._fromArrayType((ClassStack) null, genericArrayType0, (TypeBindings) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unrecognized Type: [null]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      typeFactory0.clearCache();
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LongNode> class0 = LongNode.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      assertTrue(collectionLikeType0.isContainerType());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashSet> class0 = HashSet.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      JavaType javaType0 = typeFactory0.constructSpecializedType(collectionType0, class0);
      assertEquals(1, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MapLikeType> class0 = MapLikeType.class;
      Class<ArrayType> class1 = ArrayType.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class1);
      assertFalse(collectionLikeType0.isFinal());
      
      Class<MapType> class2 = MapType.class;
      JavaType javaType0 = typeFactory0.constructSpecializedType(collectionLikeType0, class2);
      assertTrue(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<SerializationFeature> class0 = SerializationFeature.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      assertTrue(arrayType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
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
  public void test15()  throws Throwable  {
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
  public void test16()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      assertFalse(mapLikeType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      ClassLoader classLoader0 = ClassLoader.getSystemClassLoader();
      TypeFactory typeFactory1 = typeFactory0.withClassLoader(classLoader0);
      try { 
        typeFactory1.constructFromCanonical("3N|1");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Failed to parse type '3N|1' (remaining: ''): Can not locate class '3N|1', problem: 3N|1
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Object> class0 = Object.class;
      Class<SerializationFeature> class1 = SerializationFeature.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(class1, class0, typeBindings0);
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LongNode> class0 = LongNode.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructReferenceType(class0, (JavaType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.ReferenceType", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeFactory typeFactory1 = typeFactory0.withCache((LRUMap<Object, JavaType>) null);
      assertNotSame(typeFactory0, typeFactory1);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      Class<MapperFeature> class1 = MapperFeature.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      ArrayType arrayType0 = typeFactory0.constructArrayType((JavaType) collectionType0);
      assertFalse(arrayType0.isAbstract());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      LRUMap<Object, JavaType> lRUMap0 = new LRUMap<Object, JavaType>(1608, 1608);
      TypeFactory typeFactory0 = new TypeFactory(lRUMap0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TypeFactory typeFactory0 = new TypeFactory((LRUMap<Object, JavaType>) null);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeFactory typeFactory1 = typeFactory0.withModifier((TypeModifier) null);
      assertNotSame(typeFactory1, typeFactory0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      TypeModifier typeModifier1 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory2 = typeFactory1.withModifier(typeModifier1);
      assertNotSame(typeFactory1, typeFactory2);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
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
  public void test27()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      Class<?> class1 = TypeFactory.rawClass(class0);
      assertFalse(class1.isInterface());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      try { 
        typeFactory0.constructFromCanonical("unk.nown");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Failed to parse type 'unk.nown' (remaining: ''): Can not locate class 'unk.nown', problem: Class 'unk/nown.class' should be in target project, but could not be found!
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0.constructFromCanonical("char");
      assertEquals(0, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0.constructFromCanonical("int");
      assertTrue(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0.constructFromCanonical("long");
      assertTrue(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("float");
      assertEquals("float", class0.toString());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0.constructFromCanonical("double");
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("boolean");
      assertEquals("boolean", class0.toString());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("byte");
      assertEquals("byte", class0.toString());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0.constructFromCanonical("short");
      assertTrue(javaType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0.constructFromCanonical("void");
      assertFalse(javaType0.isContainerType());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0._unknownType();
      assertTrue(javaType0.isJavaLangObject());
      
      Class<MapLikeType> class0 = MapLikeType.class;
      JavaType javaType1 = typeFactory0.constructSpecializedType(javaType0, class0);
      JavaType javaType2 = typeFactory0.moreSpecificType(javaType0, javaType1);
      assertFalse(javaType2.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashSet> class0 = HashSet.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      Class<MapType> class1 = MapType.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructSpecializedType(collectionType0, class1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.type.MapType not subtype of [collection type; class java.util.HashSet, contains [collection type; class java.util.HashSet, contains [simple type, class java.lang.Object]]]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      Class<Object> class1 = Object.class;
      JavaType javaType0 = typeFactory0.constructGeneralizedType(collectionType0, class1);
      assertFalse(javaType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_COMPARABLE;
      Class<Object> class0 = Object.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructGeneralizedType(simpleType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Internal error: class java.lang.Object not included as super-type for [simple type, class java.lang.Comparable]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_COMPARABLE;
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructGeneralizedType(simpleType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.ObjectMapper$DefaultTyping not a super-type of [simple type, class java.lang.Comparable]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<String> class0 = String.class;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(class0, class0);
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0._unknownType();
      JavaType javaType1 = typeFactory0.moreSpecificType(javaType0, javaType0);
      assertTrue(javaType1.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0.moreSpecificType((JavaType) null, (JavaType) null);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      JavaType javaType0 = typeFactory0.moreSpecificType(simpleType0, (JavaType) null);
      assertFalse(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      SimpleType simpleType1 = TypeFactory.CORE_TYPE_BOOL;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0.moreSpecificType(simpleType0, simpleType1);
      assertNotSame(javaType0, simpleType1);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Properties> class0 = Properties.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (Class<?>) class0);
      assertFalse(javaType0.isCollectionLikeType());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
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
  public void test50()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      JavaType javaType0 = typeFactory0.constructType((Type) simpleType0, (JavaType) simpleType0);
      assertTrue(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Properties> class0 = Properties.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (JavaType) null);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class0);
      assertEquals(1, collectionLikeType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Properties> class0 = Properties.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructMapType((Class<? extends Map>) null, class0, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings$TypeParamStash", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MapLikeType> class0 = MapLikeType.class;
      Class<Properties> class1 = Properties.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructMapType(class1, class0, class1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Properties with 2 type parameters: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashSet> class0 = HashSet.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0);
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, javaType0);
      Class<DeserializationFeature> class1 = DeserializationFeature.class;
      JavaType[] javaTypeArray0 = new JavaType[3];
      JavaType javaType1 = typeFactory0._constructSimple(class1, typeBindings0, javaType0, javaTypeArray0);
      assertEquals(1, javaType1.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Object> class0 = Object.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      assertNotNull(javaType0);
      assertFalse(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<RuntimeException> class0 = RuntimeException.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      doReturn(simpleType0, (JavaType) null).when(typeModifier0).modifyType(any(com.fasterxml.jackson.databind.JavaType.class) , any(java.lang.reflect.Type.class) , any(com.fasterxml.jackson.databind.type.TypeBindings.class) , any(com.fasterxml.jackson.databind.type.TypeFactory.class));
      doReturn((String) null).when(typeModifier0).toString();
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      // Undeclared exception!
      try { 
        typeFactory1.constructType((Type) class0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // TypeModifier null (of type com.fasterxml.jackson.databind.type.TypeModifier$MockitoMock$626557284) return null for type [simple type, class java.io.Serializable]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<RuntimeException> class0 = RuntimeException.class;
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      doReturn((JavaType) null).when(typeModifier0).modifyType(any(com.fasterxml.jackson.databind.JavaType.class) , any(java.lang.reflect.Type.class) , any(com.fasterxml.jackson.databind.type.TypeBindings.class) , any(com.fasterxml.jackson.databind.type.TypeFactory.class));
      doReturn((String) null).when(typeModifier0).toString();
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      // Undeclared exception!
      try { 
        typeFactory1.constructType((Type) class0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // TypeModifier Mock for TypeModifier, hashCode: 1432789819 (of type com.fasterxml.jackson.databind.type.TypeModifier$MockitoMock$626557284) return null for type [simple type, class java.lang.Object]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      ObjectMapper objectMapper1 = objectMapper0.setTypeFactory(typeFactory0);
      Class<MapType> class0 = MapType.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ObjectReader objectReader0 = objectMapper1.readerFor((JavaType) simpleType0);
      // Undeclared exception!
      try { 
        objectReader0.at("JSON");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid input: JSON Pointer expression must start with '/': \"JSON\"
         //
         verifyException("com.fasterxml.jackson.core.JsonPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(simpleType0, simpleType0);
      ReferenceType referenceType1 = referenceType0.withValueHandler(typeFactory0);
      ReferenceType referenceType2 = referenceType1.withContentTypeHandler(class0);
      CollectionType collectionType0 = typeFactory0.constructCollectionType((Class<? extends Collection>) class0, (JavaType) referenceType2);
      assertTrue(collectionType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      Class<Integer> class1 = Integer.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      assertFalse(collectionType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      ParameterizedType parameterizedType0 = mock(ParameterizedType.class, new ViolatedAssumptionAnswer());
      doReturn((Type[]) null).when(parameterizedType0).getActualTypeArguments();
      doReturn((Type) null).when(parameterizedType0).getRawType();
      // Undeclared exception!
      try { 
        typeFactory0._fromParamType((ClassStack) null, parameterizedType0, (TypeBindings) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }
}
