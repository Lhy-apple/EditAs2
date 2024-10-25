/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:15:55 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.dom;

import org.junit.Test;
import static org.junit.Assert.*;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.VariablePointer;
import org.apache.commons.jxpath.ri.model.dom.DOMAttributeIterator;
import org.apache.html.dom.HTMLDocumentImpl;
import org.apache.html.dom.HTMLFrameElementImpl;
import org.apache.html.dom.HTMLStyleElementImpl;
import org.apache.wml.dom.WMLDocumentImpl;
import org.apache.wml.dom.WMLInputElementImpl;
import org.apache.wml.dom.WMLTdElementImpl;
import org.apache.xerces.dom.CoreDocumentImpl;
import org.apache.xerces.dom.DocumentTypeImpl;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DOMAttributeIterator_ESTest extends DOMAttributeIterator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      QName qName0 = new QName("");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      DocumentTypeImpl documentTypeImpl0 = new DocumentTypeImpl((CoreDocumentImpl) null, "<<unknown namespace>>");
      WMLDocumentImpl wMLDocumentImpl0 = new WMLDocumentImpl(documentTypeImpl0);
      WMLTdElementImpl wMLTdElementImpl0 = new WMLTdElementImpl(wMLDocumentImpl0, "");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, qName0, wMLTdElementImpl0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      int int0 = dOMAttributeIterator0.getPosition();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      DocumentTypeImpl documentTypeImpl0 = new DocumentTypeImpl((CoreDocumentImpl) null, "*");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer((NodePointer) null, qName0, documentTypeImpl0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      DocumentTypeImpl documentTypeImpl0 = new DocumentTypeImpl((CoreDocumentImpl) null, "*");
      WMLDocumentImpl wMLDocumentImpl0 = new WMLDocumentImpl(documentTypeImpl0);
      WMLInputElementImpl wMLInputElementImpl0 = new WMLInputElementImpl(wMLDocumentImpl0, "*");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer((NodePointer) null, qName0, wMLInputElementImpl0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
      
      dOMAttributeIterator0.getNodePointer();
      assertEquals(1, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      QName qName0 = new QName("xmlns");
      DocumentTypeImpl documentTypeImpl0 = new DocumentTypeImpl((CoreDocumentImpl) null, "xmlns");
      WMLDocumentImpl wMLDocumentImpl0 = new WMLDocumentImpl(documentTypeImpl0);
      WMLInputElementImpl wMLInputElementImpl0 = new WMLInputElementImpl(wMLDocumentImpl0, "xmlns");
      wMLInputElementImpl0.setAttribute("xmlns", "xmlns");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer((NodePointer) null, qName0, wMLInputElementImpl0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      DocumentTypeImpl documentTypeImpl0 = new DocumentTypeImpl((CoreDocumentImpl) null, "*");
      WMLDocumentImpl wMLDocumentImpl0 = new WMLDocumentImpl(documentTypeImpl0);
      WMLInputElementImpl wMLInputElementImpl0 = new WMLInputElementImpl(wMLDocumentImpl0, "*");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer((NodePointer) null, qName0, wMLInputElementImpl0);
      wMLInputElementImpl0.setType("http://www.w3.org/2000/xmlns/");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      NodePointer nodePointer1 = dOMAttributeIterator0.getNodePointer();
      assertNotNull(nodePointer1);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      QName qName0 = new QName("xmlns", "xmlns");
      DocumentTypeImpl documentTypeImpl0 = new DocumentTypeImpl((CoreDocumentImpl) null, "xmlns");
      WMLDocumentImpl wMLDocumentImpl0 = new WMLDocumentImpl(documentTypeImpl0);
      WMLInputElementImpl wMLInputElementImpl0 = new WMLInputElementImpl(wMLDocumentImpl0, "xmlns");
      wMLInputElementImpl0.setAttribute("xmlns", "xmlns");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer((NodePointer) null, qName0, wMLInputElementImpl0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      QName qName0 = new QName("xmlns", "xmlns");
      DocumentTypeImpl documentTypeImpl0 = new DocumentTypeImpl((CoreDocumentImpl) null, "xmlns");
      WMLDocumentImpl wMLDocumentImpl0 = new WMLDocumentImpl(documentTypeImpl0);
      WMLInputElementImpl wMLInputElementImpl0 = new WMLInputElementImpl(wMLDocumentImpl0, "xmlns");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer((NodePointer) null, qName0, wMLInputElementImpl0);
      wMLInputElementImpl0.setTitle("http://www.w3.org/XML/1998/namespace");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      QName qName0 = new QName("*");
      DocumentTypeImpl documentTypeImpl0 = new DocumentTypeImpl((CoreDocumentImpl) null, "*");
      WMLDocumentImpl wMLDocumentImpl0 = new WMLDocumentImpl(documentTypeImpl0);
      WMLInputElementImpl wMLInputElementImpl0 = new WMLInputElementImpl(wMLDocumentImpl0, "*");
      wMLInputElementImpl0.setValue("*");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer((NodePointer) null, qName0, wMLInputElementImpl0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      QName qName0 = new QName("*");
      DocumentTypeImpl documentTypeImpl0 = new DocumentTypeImpl((CoreDocumentImpl) null, "*");
      WMLDocumentImpl wMLDocumentImpl0 = new WMLDocumentImpl(documentTypeImpl0);
      WMLInputElementImpl wMLInputElementImpl0 = new WMLInputElementImpl(wMLDocumentImpl0, "*");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer((NodePointer) null, qName0, wMLInputElementImpl0);
      wMLInputElementImpl0.setXmlLang("http://www.w3.org/2000/xmlns/");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      QName qName0 = new QName("xmlns", "xmlns");
      DocumentTypeImpl documentTypeImpl0 = new DocumentTypeImpl((CoreDocumentImpl) null, "xmlns");
      WMLDocumentImpl wMLDocumentImpl0 = new WMLDocumentImpl(documentTypeImpl0);
      WMLInputElementImpl wMLInputElementImpl0 = new WMLInputElementImpl(wMLDocumentImpl0, "xmlns");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer((NodePointer) null, qName0, wMLInputElementImpl0);
      HTMLFrameElementImpl hTMLFrameElementImpl0 = new HTMLFrameElementImpl((HTMLDocumentImpl) null, "http://www.w3.org/XML/1998/namespace");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) hTMLFrameElementImpl0);
      NodePointer nodePointer1 = nodePointer0.createAttribute(jXPathContext0, qName0);
      assertFalse(nodePointer1.isCollection());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      QName qName0 = new QName(",", ",");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLStyleElementImpl hTMLStyleElementImpl0 = new HTMLStyleElementImpl(hTMLDocumentImpl0, "<<unknown namespace>>");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, qName0, hTMLStyleElementImpl0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      boolean boolean0 = dOMAttributeIterator0.setPosition(Integer.MIN_VALUE);
      assertEquals(Integer.MIN_VALUE, dOMAttributeIterator0.getPosition());
      assertFalse(boolean0);
  }
}
